"""Doppler-only analysis for CARLA radar point-cloud datasets.

This tool reads the timestamped ``radar_spher/*.npy`` files produced by
``new_delft2.py``. It does not use closed-loop metrics or a trained model.
The optional ``radar_4d_spher`` directory is compared with native radar at
matched timestamps. Existing PD/PFA CSV summaries can be included as context,
but Doppler statistics are computed from the point files themselves.

Example:
    python3 radar_pipeline/doppler_dataset_analysis.py \
        --root ./Town02_3d_radar \
        --native-pd-pfa ./native_3d.csv \
        --modified-pd-pfa ./modified_4d.csv \
        --json ./doppler_summary.json
"""

import argparse
import csv
import json
import math
import os
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

ROI = (0.0, -70.0, -15.0, 50.0, 70.0, 15.0)


def dataset_dir(root: str) -> str:
    candidate = os.path.join(root, "radelf_spherical")
    return candidate if os.path.isdir(candidate) else root


def frame_files(root: str, subdir: str) -> List[Tuple[str, str]]:
    directory = os.path.join(dataset_dir(root), subdir)
    paths = glob(os.path.join(directory, "*.npy"))
    return sorted((os.path.basename(path), path) for path in paths)


def select_frames(
    files: List[Tuple[str, str]], split: str, max_files: Optional[int]
) -> List[Tuple[str, str]]:
    if split != "all":
        start_fraction = {"train": 0.0, "val": 0.8, "test": 0.9}[split]
        end_fraction = {"train": 0.8, "val": 0.9, "test": 1.0}[split]
        start = int(len(files) * start_fraction)
        end = int(len(files) * end_fraction)
        files = files[start:end]
    return files[:max_files] if max_files is not None else files


def load_points(path: str) -> np.ndarray:
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected (N,4+) radar points in {path}, got {points.shape}")
    points = points[:, :4].astype(np.float64, copy=False)
    r_min, az_min, el_min, r_max, az_max, el_max = ROI
    mask = (
        (points[:, 0] >= r_min) & (points[:, 0] <= r_max)
        & (points[:, 1] >= np.deg2rad(az_min))
        & (points[:, 1] <= np.deg2rad(az_max))
        & (points[:, 2] >= np.deg2rad(el_min))
        & (points[:, 2] <= np.deg2rad(el_max))
        & np.isfinite(points).all(axis=1)
    )
    return points[mask]


def velocity_stats(points: np.ndarray) -> Dict[str, float]:
    velocity = points[:, 3] if len(points) else np.empty(0)
    if not len(velocity):
        return {
            "detections": 0.0, "mean_ms": float("nan"), "std_ms": float("nan"),
            "min_ms": float("nan"), "max_ms": float("nan"),
            "approaching_fraction": float("nan"), "receding_fraction": float("nan"),
            "near_zero_fraction": float("nan"),
        }
    return {
        "detections": float(len(velocity)),
        "mean_ms": float(velocity.mean()),
        "std_ms": float(velocity.std()),
        "min_ms": float(velocity.min()),
        "max_ms": float(velocity.max()),
        "approaching_fraction": float(np.mean(velocity > 0.1)),
        "receding_fraction": float(np.mean(velocity < -0.1)),
        "near_zero_fraction": float(np.mean(np.abs(velocity) <= 0.1)),
    }


def frame_time(name: str) -> float:
    return float(os.path.splitext(name)[0])


def nearest_matches(
    source: np.ndarray,
    target: np.ndarray,
    range_tolerance: float,
    angle_tolerance: float,
    max_source: int = 1500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match points by normalized range/angle/elevation distance."""
    if not len(source) or not len(target):
        return np.empty(0), np.empty(0, dtype=np.int64)
    if len(source) > max_source:
        source = source[np.linspace(0, len(source) - 1, max_source).astype(int)]
    target_features = target[:, :3].copy()
    source_features = source[:, :3].copy()
    target_features[:, 0] /= range_tolerance
    target_features[:, 1:] /= angle_tolerance
    source_features[:, 0] /= range_tolerance
    source_features[:, 1:] /= angle_tolerance
    distances, indices = cKDTree(target_features).query(
        source_features, k=4, distance_upper_bound=1.0
    )
    candidates = []
    for source_index in range(len(source)):
        for candidate_distance, target_index in zip(
            np.atleast_1d(distances[source_index]),
            np.atleast_1d(indices[source_index]),
        ):
            if np.isfinite(candidate_distance) and target_index < len(target):
                candidates.append((float(candidate_distance), source_index, int(target_index)))
    candidates.sort()
    used_sources = set()
    used_targets = set()
    selected_sources = []
    selected_targets = []
    for _, source_index, target_index in candidates:
        if source_index in used_sources or target_index in used_targets:
            continue
        used_sources.add(source_index)
        used_targets.add(target_index)
        selected_sources.append(source_index)
        selected_targets.append(target_index)
    return source[selected_sources], np.asarray(selected_targets, dtype=np.int64)


def temporal_motion_analysis(
    frames: List[Tuple[str, str]],
    range_tolerance: float,
    angle_tolerance: float,
) -> Dict[str, float]:
    measured = []
    range_rate = []
    transitions = 0
    for (name_a, path_a), (name_b, path_b) in zip(frames, frames[1:]):
        dt = frame_time(name_b) - frame_time(name_a)
        if dt <= 0:
            continue
        points_a = load_points(path_a)
        points_b = load_points(path_b)
        matched, indices = nearest_matches(
            points_a, points_b, range_tolerance, angle_tolerance
        )
        if not len(matched):
            continue
        transitions += len(matched)
        delta_range_rate = (points_b[indices, 0] - matched[:, 0]) / dt
        measured.extend(matched[:, 3])
        range_rate.extend(delta_range_rate)
    measured = np.asarray(measured, dtype=np.float64)
    range_rate = np.asarray(range_rate, dtype=np.float64)
    if not len(measured):
        return {"matches": 0.0}
    approaching_truth = -range_rate
    receding_truth = range_rate

    def compare(truth: np.ndarray) -> Dict[str, float]:
        error = measured - truth
        return {
            "bias_ms": float(error.mean()),
            "mae_ms": float(np.abs(error).mean()),
            "rmse_ms": float(np.sqrt(np.mean(error ** 2))),
            "sign_agreement": float(np.mean(np.sign(measured) == np.sign(truth))),
            "correlation": float(np.corrcoef(measured, truth)[0, 1])
            if np.std(measured) > 0 and np.std(truth) > 0 else float("nan"),
        }

    return {
        "matches": float(transitions),
        "mean_native_ms": float(measured.mean()),
        "mean_range_rate_ms": float(range_rate.mean()),
        "native_vs_approaching_convention": compare(approaching_truth),
        "native_vs_receding_convention": compare(receding_truth),
    }


def compare_variants(
    native_frames: List[Tuple[str, str]],
    modified_root: str,
    range_tolerance: float,
    angle_tolerance: float,
) -> Dict[str, float]:
    modified_by_name = dict(frame_files(modified_root, "radar_4d_spher"))
    deltas = []
    native_velocities = []
    modified_velocities = []
    frame_means = []
    frame_maes = []
    frame_rmses = []
    frame_match_rates = []
    range_values = []
    for name, native_path in native_frames:
        modified_path = modified_by_name.get(name)
        if modified_path is None:
            continue
        native = load_points(native_path)
        modified = load_points(modified_path)
        matched, indices = nearest_matches(
            modified, native, range_tolerance, angle_tolerance
        )
        if len(matched):
            frame_delta = matched[:, 3] - native[indices, 3]
            deltas.extend(frame_delta)
            native_velocities.extend(native[indices, 3])
            modified_velocities.extend(matched[:, 3])
            range_values.extend(matched[:, 0])
            frame_means.append(float(frame_delta.mean()))
            frame_maes.append(float(np.abs(frame_delta).mean()))
            frame_rmses.append(float(np.sqrt(np.mean(frame_delta ** 2))))
            frame_match_rates.append(float(len(matched) / max(len(modified), 1)))
    deltas = np.asarray(deltas, dtype=np.float64)
    if not len(deltas):
        return {"matches": 0.0}
    native_velocities = np.asarray(native_velocities, dtype=np.float64)
    modified_velocities = np.asarray(modified_velocities, dtype=np.float64)
    range_values = np.asarray(range_values, dtype=np.float64)

    def frame_summary(values: List[float]) -> Dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(20261007)
        bootstrap_means = np.empty(1000, dtype=np.float64)
        for bootstrap_index in range(len(bootstrap_means)):
            sample = rng.choice(values, size=len(values), replace=True)
            bootstrap_means[bootstrap_index] = sample.mean()
        return {
            "frames": float(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "bootstrap_ci95_low": float(np.percentile(bootstrap_means, 2.5)),
            "bootstrap_ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        }

    range_summary = {}
    for lower, upper in ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.0)):
        mask = (range_values >= lower) & (range_values < upper)
        range_deltas = deltas[mask]
        if len(range_deltas):
            range_summary[f"{int(lower)}_{int(upper)}m"] = {
                "samples": float(len(range_deltas)),
                "bias_ms": float(range_deltas.mean()),
                "mae_ms": float(np.abs(range_deltas).mean()),
                "rmse_ms": float(np.sqrt(np.mean(range_deltas ** 2))),
            }

    return {
        "matches": float(len(deltas)),
        "matched_frame_count": float(len(frame_means)),
        "mean_match_rate": float(np.mean(frame_match_rates)),
        "frame_mean_delta": frame_summary(frame_means),
        "frame_mae": frame_summary(frame_maes),
        "frame_rmse": frame_summary(frame_rmses),
        "mean_modified_minus_native_ms": float(deltas.mean()),
        "mae_ms": float(np.abs(deltas).mean()),
        "rmse_ms": float(np.sqrt(np.mean(deltas ** 2))),
        "std_ms": float(deltas.std()),
        "p95_absolute_error_ms": float(np.percentile(np.abs(deltas), 95)),
        "wasserstein_velocity_distance_ms": float(
            wasserstein_distance(native_velocities, modified_velocities)
        ),
        "range_stratified": range_summary,
    }


def read_pd_pfa(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as input_file:
        row = next(csv.DictReader(input_file))
    return {key: float(value) for key, value in row.items()}


def aggregate_stats(frame_stats: List[Dict[str, float]]) -> Dict[str, float]:
    if not frame_stats:
        return {"frames": 0.0}
    result = {"frames": float(len(frame_stats))}
    for key in frame_stats[0]:
        values = np.asarray([item[key] for item in frame_stats], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite):
            result[f"{key}_mean"] = float(finite.mean())
            result[f"{key}_std"] = float(finite.std())
    return result


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    native_frames = select_frames(
        frame_files(args.root, "radar_spher"), args.split, args.max_files
    )
    if not native_frames:
        raise RuntimeError(f"No native radar files found under {args.root!r}")
    native_stats = aggregate_stats([velocity_stats(load_points(path)) for _, path in native_frames])
    result = {
        "dataset_root": args.root,
        "split": args.split,
        "native": native_stats,
        "native_temporal_motion": temporal_motion_analysis(
            native_frames, args.range_tolerance, args.angle_tolerance
        ),
        "native_pd_pfa": read_pd_pfa(args.native_pd_pfa),
        "modified_pd_pfa": read_pd_pfa(args.modified_pd_pfa),
    }
    if args.modified_root:
        modified_frames = select_frames(
            frame_files(args.modified_root, "radar_4d_spher"),
            args.split,
            args.max_files,
        )
        result["modified"] = aggregate_stats(
            [velocity_stats(load_points(path)) for _, path in modified_frames]
        )
        result["modified_vs_native"] = compare_variants(
            native_frames, args.modified_root,
            args.range_tolerance, args.angle_tolerance,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CARLA Doppler without closed-loop metrics.")
    parser.add_argument("--root", required=True, help="Dataset root or radelf_spherical directory")
    parser.add_argument("--modified-root", default=None, help="Dataset containing radar_4d_spher")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--range-tolerance", type=float, default=0.75)
    parser.add_argument("--angle-tolerance", type=float, default=0.035)
    parser.add_argument("--native-pd-pfa", default=None)
    parser.add_argument("--modified-pd-pfa", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.range_tolerance <= 0 or args.angle_tolerance <= 0:
        parser.error("matching tolerances must be positive")
    summary = analyze(args)
    print(json.dumps(summary, indent=2, allow_nan=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, allow_nan=True)
            output_file.write("\n")


if __name__ == "__main__":
    main()
