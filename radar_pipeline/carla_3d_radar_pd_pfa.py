"""Offline CARLA native 3D-radar baseline for PD/PFA evaluation.

The CARLA collector in ``new_delft2.py`` writes native radar detections as
``[range_m, azimuth_rad, elevation_rad, radial_velocity_mps]`` and LiDAR
points in the same spherical coordinate convention.  This script evaluates
the native radar occupancy directly on the same range/azimuth grid used by
the 4D detector:

    python3 carla_3d_radar_pd_pfa.py --root ./training_data --split test

It does not load a trained detector or start CARLA.  Use ``--export-dir`` to
save binary 3D-radar occupancy and collapsed LiDAR target masks for later
training experiments.  A JSON ``RadarPhysicsConfig`` can be supplied with
``--physics-config`` to evaluate noise, multipath, clutter, and range-dependent
detection loss without CARLA.
"""

import argparse
import csv
import json
import os
from glob import glob
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from radar_pipeline.radar_physics import RadarPhysicsConfig, apply_radar_physics
except ModuleNotFoundError:
    from radar_physics import RadarPhysicsConfig, apply_radar_physics


# Match get_default_params() and CarlaRADCUBE_DATASET_TIME exactly.
ROI_SPHERICAL = (0.0, -70.0, -15.0, 50.0, 70.0, 15.0)
RANGE_CELL_SIZE = 0.1004
MAX_RANGE = 51.4242
RANGE_AXIS = np.arange(RANGE_CELL_SIZE, MAX_RANGE + RANGE_CELL_SIZE,
                       RANGE_CELL_SIZE)[10:-3]

_wx = np.linspace(-np.pi, np.pi, 256)[8:248]
AZIMUTH_AXIS = np.arcsin(_wx / (2 * np.pi * 0.4972))


def crop_to_roi(points: np.ndarray) -> np.ndarray:
    """Return points inside the training radar spherical ROI."""
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


def points_to_ra_mask(points: np.ndarray) -> np.ndarray:
    """Project spherical points into the (range, azimuth) occupancy grid."""
    mask = np.zeros((len(RANGE_AXIS), len(AZIMUTH_AXIS)), dtype=bool)
    points = crop_to_roi(points)
    if points.size == 0:
        return mask

    range_idx = np.searchsorted(RANGE_AXIS, points[:, 0], side="left")
    azimuth_idx = np.searchsorted(AZIMUTH_AXIS, points[:, 1], side="left")
    valid = (
        (range_idx >= 0) & (range_idx < len(RANGE_AXIS))
        & (azimuth_idx >= 0) & (azimuth_idx < len(AZIMUTH_AXIS))
    )
    mask[range_idx[valid], azimuth_idx[valid]] = True
    return mask


def pd_pfa(target: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
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


def split_files(
    root: str, split: str, radar_subdir: str = "radar_spher"
) -> Iterable[Tuple[str, str]]:
    """Yield matching radar/LiDAR files for the requested temporal split."""
    radar_dir = os.path.join(root, radar_subdir)
    lidar_dir = os.path.join(root, "lidar_spher")
    radar = {os.path.basename(path): path
             for path in glob(os.path.join(radar_dir, "*.npy"))}
    lidar = {os.path.basename(path): path
             for path in glob(os.path.join(lidar_dir, "*.npy"))}
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
    split: str,
    min_velocity: Optional[float],
    min_points: int,
    max_files: Optional[int],
    export_dir: Optional[str],
    physics_config: Optional[RadarPhysicsConfig] = None,
    radar_subdir: str = "radar_spher",
) -> Dict[str, float]:
    """Evaluate native radar occupancy and optionally export baseline masks."""
    files = list(split_files(root, split, radar_subdir))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(
            f"No matching radar_spher/lidar_spher files found in {root!r}."
        )

    if export_dir:
        os.makedirs(os.path.join(export_dir, "radar_3d_ra"), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "lidar_target_ra"), exist_ok=True)

    frame_metrics = []
    for radar_path, lidar_path in files:
        radar = np.load(radar_path)
        lidar = np.load(lidar_path)
        if radar.ndim != 2 or radar.shape[1] < 4:
            raise ValueError(f"Expected radar shape (N,4+), got {radar.shape}")
        if lidar.ndim != 2 or lidar.shape[1] < 3:
            raise ValueError(f"Expected LiDAR shape (N,3+), got {lidar.shape}")

        radar = radar[:, :4].astype(np.float32, copy=False)
        lidar = lidar[:, :4].astype(np.float32, copy=False)
        if physics_config is not None:
            radar = apply_radar_physics(radar, physics_config)
        radar = crop_to_roi(radar)
        if min_velocity is not None:
            radar = radar[np.abs(radar[:, 3]) >= min_velocity]

        # A native radar cell can contain multiple detections.  min_points
        # exposes a deterministic strength knob without pretending the
        # simulator provides a learned confidence score.
        radar_mask = points_to_ra_mask(radar)
        if min_points > 1 and radar.size:
            ranges = np.searchsorted(RANGE_AXIS, radar[:, 0], side="left")
            azimuths = np.searchsorted(AZIMUTH_AXIS, radar[:, 1], side="left")
            valid = (
                (ranges >= 0) & (ranges < len(RANGE_AXIS))
                & (azimuths >= 0) & (azimuths < len(AZIMUTH_AXIS))
            )
            counts = np.zeros_like(radar_mask, dtype=np.int32)
            np.add.at(counts, (ranges[valid], azimuths[valid]), 1)
            radar_mask = counts >= min_points

        target_mask = points_to_ra_mask(lidar)
        pd, pfa = pd_pfa(target_mask, radar_mask)
        frame_metrics.append((pd, pfa))

        if export_dir:
            name = os.path.basename(radar_path)
            np.save(os.path.join(export_dir, "radar_3d_ra", name),
                    radar_mask.astype(np.uint8))
            np.save(os.path.join(export_dir, "lidar_target_ra", name),
                    target_mask.astype(np.uint8))

    values = np.asarray(frame_metrics, dtype=np.float64)
    summary = {
        "frames": float(len(values)),
        "pd_mean": float(values[:, 0].mean()),
        "pfa_mean": float(values[:, 1].mean()),
        "pd_std": float(values[:, 0].std()),
        "pfa_std": float(values[:, 1].std()),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute PD/PFA for native CARLA 3D radar."
    )
    parser.add_argument("--root", required=True,
                        help="Dataset root containing radar_spher/ and lidar_spher/.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"),
                        default="test")
    parser.add_argument("--min-velocity", type=float, default=None,
                        help="Keep radar detections with abs(velocity) >= m/s.")
    parser.add_argument("--min-points", type=int, default=1,
                        help="Minimum detections per range/azimuth cell.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--radar-subdir",
        default="radar_spher",
        help="Radar directory under ROOT (use radar_4d_spher for modified 4D data).",
    )
    parser.add_argument("--export-dir", default=None,
                        help="Optional directory for 3D radar/target RA masks.")
    parser.add_argument("--csv", default=None,
                        help="Optional path for per-run summary CSV.")
    parser.add_argument(
        "--physics-config",
        default=None,
        help="Optional JSON file containing RadarPhysicsConfig fields.",
    )
    args = parser.parse_args()

    if args.min_points < 1:
        parser.error("--min-points must be at least 1")
    physics_config = None
    if args.physics_config:
        with open(args.physics_config, encoding="utf-8") as config_file:
            config_values = json.load(config_file)
        config_values["enabled"] = True
        physics_config = RadarPhysicsConfig(**config_values)
    summary = evaluate(
        root=args.root,
        split=args.split,
        min_velocity=args.min_velocity,
        min_points=args.min_points,
        max_files=args.max_files,
        export_dir=args.export_dir,
        physics_config=physics_config,
        radar_subdir=args.radar_subdir,
    )
    print(f"CARLA radar occupancy ({args.radar_subdir}, {args.split})")
    print(f"  Frames : {int(summary['frames'])}")
    print(f"  Pd     : {summary['pd_mean']:.6f} +/- {summary['pd_std']:.6f}")
    print(f"  Pfa    : {summary['pfa_mean']:.6f} +/- {summary['pfa_std']:.6f}")
    if args.csv:
        with open(args.csv, "w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(summary.keys())
            writer.writerow(summary.values())


if __name__ == "__main__":
    main()
