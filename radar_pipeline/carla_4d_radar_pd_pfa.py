"""Offline CARLA native 4D-radar PD/PFA evaluation.

Native 4D radar points are stored as
``[range_m, azimuth_rad, elevation_rad, radial_velocity_mps]``.
The additional velocity coordinate is discretized into a Doppler axis before
the occupancy is reduced over Doppler for comparison with the LiDAR target.
This keeps the PD/PFA definition identical to the 3D baseline while retaining
the 4D radar representation during projection.
"""

import argparse
import csv
import os
from glob import glob
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

ROI_SPHERICAL = (0.0, -70.0, -15.0, 50.0, 70.0, 15.0)
RANGE_CELL_SIZE = 0.1004
MAX_RANGE = 51.4242
RANGE_AXIS = np.arange(
    RANGE_CELL_SIZE, MAX_RANGE + RANGE_CELL_SIZE, RANGE_CELL_SIZE
)[10:-3]
_wx = np.linspace(-np.pi, np.pi, 256)[8:248]
AZIMUTH_AXIS = np.arcsin(_wx / (2 * np.pi * 0.4972))

# Match get_default_params() / CarlaRADCUBE_DATASET_TIME.
DOPPLER_BINS = 128
DOPPLER_BIN_SIZE = 0.04607058455831936


def compute_pd_pfa(target: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    """Compute cell-level probability of detection and false alarm."""
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError(
            f"Target/prediction shapes differ: {target.shape} vs {prediction.shape}"
        )
    positives = int(target.sum())
    negatives = int((~target).sum())
    true_positives = int((target & prediction).sum())
    false_positives = int((~target & prediction).sum())
    return (
        true_positives / positives if positives else 0.0,
        false_positives / negatives if negatives else 0.0,
    )


def crop_to_roi(points: np.ndarray) -> np.ndarray:
    if points is None or points.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    r_min, az_min, el_min, r_max, az_max, el_max = ROI_SPHERICAL
    mask = (
        (points[:, 0] >= r_min) & (points[:, 0] <= r_max)
        & (points[:, 1] >= np.deg2rad(az_min))
        & (points[:, 1] <= np.deg2rad(az_max))
        & (points[:, 2] >= np.deg2rad(el_min))
        & (points[:, 2] <= np.deg2rad(el_max))
    )
    return points[mask].astype(np.float32, copy=False)


def points_to_4d_mask(points: np.ndarray) -> np.ndarray:
    """Project points into a Doppler, range, azimuth occupancy cube."""
    mask = np.zeros(
        (DOPPLER_BINS, len(RANGE_AXIS), len(AZIMUTH_AXIS)), dtype=bool
    )
    points = crop_to_roi(points)
    if points.size == 0:
        return mask

    doppler = np.rint(
        points[:, 3] / DOPPLER_BIN_SIZE + DOPPLER_BINS / 2.0
    ).astype(np.int64)
    ranges = np.searchsorted(RANGE_AXIS, points[:, 0], side="left")
    azimuths = np.searchsorted(AZIMUTH_AXIS, points[:, 1], side="left")
    valid = (
        (doppler >= 0) & (doppler < DOPPLER_BINS)
        & (ranges >= 0) & (ranges < len(RANGE_AXIS))
        & (azimuths >= 0) & (azimuths < len(AZIMUTH_AXIS))
    )
    mask[doppler[valid], ranges[valid], azimuths[valid]] = True
    return mask


def split_files(
    root: str, split: str, radar_subdir: str = "radar_4d_spher"
) -> Iterable[Tuple[str, str]]:
    radar_dir = os.path.join(root, radar_subdir)
    lidar_dir = os.path.join(root, "lidar_spher")
    radar = {os.path.basename(p): p for p in glob(os.path.join(radar_dir, "*.npy"))}
    lidar = {os.path.basename(p): p for p in glob(os.path.join(lidar_dir, "*.npy"))}
    names = sorted(set(radar) & set(lidar))
    if split == "all":
        selected = names
    else:
        start = {"train": 0.0, "val": 0.8, "test": 0.9}[split]
        end = {"train": 0.8, "val": 0.9, "test": 1.0}[split]
        selected = names[int(len(names) * start):int(len(names) * end)]
    return ((radar[name], lidar[name]) for name in selected)


def evaluate(
    root: str,
    split: str = "test",
    min_velocity: Optional[float] = None,
    min_points: int = 1,
    max_files: Optional[int] = None,
    radar_subdir: str = "radar_4d_spher",
    physics_config=None,
) -> Dict[str, float]:
    files = list(split_files(root, split, radar_subdir))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(
            f"No matching {radar_subdir}/lidar_spher files found in {root!r}."
        )

    frame_metrics = []
    for radar_path, lidar_path in files:
        radar = np.load(radar_path)
        lidar = np.load(lidar_path)
        if radar.ndim != 2 or radar.shape[1] < 4:
            raise ValueError(f"Expected radar shape (N,4+), got {radar.shape}")
        if lidar.ndim != 2 or lidar.shape[1] < 3:
            raise ValueError(f"Expected LiDAR shape (N,3+), got {lidar.shape}")

        radar = radar[:, :4].astype(np.float32, copy=False)
        lidar = np.concatenate(
            (
                lidar[:, :3].astype(np.float32, copy=False),
                np.zeros((len(lidar), 1), dtype=np.float32),
            ),
            axis=1,
        )
        if physics_config is not None:
            try:
                from radar_pipeline.radar_physics import apply_radar_physics
            except ModuleNotFoundError:
                from radar_physics import apply_radar_physics
            radar = apply_radar_physics(radar, physics_config)
        radar = crop_to_roi(radar)
        if min_velocity is not None:
            radar = radar[np.abs(radar[:, 3]) >= min_velocity]

        radar_4d = points_to_4d_mask(radar)
        if min_points > 1 and radar.size:
            doppler = np.rint(
                radar[:, 3] / DOPPLER_BIN_SIZE + DOPPLER_BINS / 2.0
            ).astype(np.int64)
            ranges = np.searchsorted(RANGE_AXIS, radar[:, 0], side="left")
            azimuths = np.searchsorted(AZIMUTH_AXIS, radar[:, 1], side="left")
            valid = (
                (doppler >= 0) & (doppler < DOPPLER_BINS)
                & (ranges >= 0) & (ranges < len(RANGE_AXIS))
                & (azimuths >= 0) & (azimuths < len(AZIMUTH_AXIS))
            )
            counts = np.zeros_like(radar_4d, dtype=np.int32)
            np.add.at(counts, (doppler[valid], ranges[valid], azimuths[valid]), 1)
            radar_4d = counts >= min_points

        # LiDAR has no Doppler coordinate. Compare spatial occupancy after
        # retaining the 4D radar cube through the projection step.
        radar_ra = radar_4d.any(axis=0)
        lidar_ra = points_to_4d_mask(lidar).any(axis=0)
        frame_metrics.append(compute_pd_pfa(lidar_ra, radar_ra))

    values = np.asarray(frame_metrics, dtype=np.float64)
    return {
        "frames": float(len(values)),
        "pd_mean": float(values[:, 0].mean()),
        "pfa_mean": float(values[:, 1].mean()),
        "pd_std": float(values[:, 0].std()),
        "pfa_std": float(values[:, 1].std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PD/PFA for native CARLA 4D radar.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--radar-subdir", default="radar_4d_spher")
    parser.add_argument("--min-velocity", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    if args.min_points < 1:
        parser.error("--min-points must be at least 1")
    summary = evaluate(
        args.root, args.split, args.min_velocity, args.min_points,
        args.max_files, args.radar_subdir,
    )
    print(f"CARLA 4D radar occupancy ({args.radar_subdir}, {args.split})")
    print(f"  Frames : {int(summary['frames'])}")
    print(f"  Pd     : {summary['pd_mean']:.6f} +/- {summary['pd_std']:.6f}")
    print(f"  Pfa    : {summary['pfa_mean']:.6f} +/- {summary['pfa_std']:.6f}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(summary.keys())
            writer.writerow(summary.values())


if __name__ == "__main__":
    main()
