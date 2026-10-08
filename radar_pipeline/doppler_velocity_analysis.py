"""Analyze preserved Doppler velocity from a closed-loop CARLA run.

The input is ``closed_loop_metrics.csv`` produced by closed_loop_runner.py.
The analysis keeps the Doppler coordinate signed and reports relationships
between native radar measurements, ego-only motion, and the lead vehicle's
exact line-of-sight radial velocity.

Example:
    python3 radar_pipeline/doppler_velocity_analysis.py \
        --input ./closed_loop_output/closed_loop_metrics.csv
"""

import argparse
import csv
import json
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np


def finite_column(rows: Iterable[Dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def paired_columns(
    rows: List[Dict[str, str]], measured_name: str, truth_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    measured = []
    truth = []
    for row in rows:
        try:
            measured_value = float(row[measured_name])
            truth_value = float(row[truth_name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(measured_value) and math.isfinite(truth_value):
            measured.append(measured_value)
            truth.append(truth_value)
    return np.asarray(measured, dtype=np.float64), np.asarray(truth, dtype=np.float64)


def comparison(measured: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    if len(measured) == 0:
        return {"samples": 0.0}
    error = measured - truth
    result = {
        "samples": float(len(measured)),
        "bias_ms": float(error.mean()),
        "mae_ms": float(np.abs(error).mean()),
        "rmse_ms": float(np.sqrt(np.mean(error ** 2))),
        "sign_agreement": float(np.mean(np.sign(measured) == np.sign(truth))),
    }
    if len(measured) > 1 and np.std(measured) > 0 and np.std(truth) > 0:
        result["correlation"] = float(np.corrcoef(measured, truth)[0, 1])
    else:
        result["correlation"] = float("nan")
    return result


def temporal_jitter(values: np.ndarray) -> Dict[str, float]:
    if len(values) < 2:
        return {"transitions": 0.0}
    delta = np.diff(values)
    return {
        "transitions": float(len(delta)),
        "mean_abs_delta_ms": float(np.abs(delta).mean()),
        "std_delta_ms": float(delta.std()),
        "sign_change_rate": float(np.mean(np.sign(values[1:]) != np.sign(values[:-1]))),
    }


def analyze(rows: List[Dict[str, str]]) -> Dict[str, object]:
    native_mean = finite_column(rows, "native_doppler_mean_ms")
    ego_mean = finite_column(rows, "ego_only_doppler_mean_ms")
    detected, ground_truth = paired_columns(
        rows, "detected_vel_ms", "gt_radial_vel_ms"
    )
    native_vs_ego, native_vs_ego_truth = paired_columns(
        rows, "native_doppler_mean_ms", "ego_only_doppler_mean_ms"
    )

    native_counts = finite_column(rows, "native_doppler_count")
    native_std = finite_column(rows, "native_doppler_std_ms")
    native_min = finite_column(rows, "native_doppler_min_ms")
    native_max = finite_column(rows, "native_doppler_max_ms")

    return {
        "frames": float(len(rows)),
        "frames_with_native_doppler": float(len(native_mean)),
        "native_detection_count": float(native_counts.sum()),
        "native_doppler": {
            "mean_of_frame_means_ms": float(native_mean.mean()) if len(native_mean) else float("nan"),
            "std_of_frame_means_ms": float(native_mean.std()) if len(native_mean) else float("nan"),
            "mean_within_frame_std_ms": float(native_std.mean()) if len(native_std) else float("nan"),
            "minimum_ms": float(native_min.min()) if len(native_min) else float("nan"),
            "maximum_ms": float(native_max.max()) if len(native_max) else float("nan"),
            "temporal": temporal_jitter(native_mean),
        },
        "native_vs_ego_only": comparison(native_vs_ego, native_vs_ego_truth),
        "detected_velocity_vs_ground_truth": comparison(detected, ground_truth),
        "detected_velocity_temporal": temporal_jitter(detected),
    }


def print_report(summary: Dict[str, object]) -> None:
    native = summary["native_doppler"]
    print("Doppler velocity study")
    print(f"  Frames                         : {int(summary['frames'])}")
    print(f"  Frames with native Doppler     : {int(summary['frames_with_native_doppler'])}")
    print(f"  Native detections              : {int(summary['native_detection_count'])}")
    print(f"  Native frame-mean velocity     : {native['mean_of_frame_means_ms']:.3f} m/s")
    print(f"  Native frame-mean variability   : {native['std_of_frame_means_ms']:.3f} m/s")
    print(f"  Native within-frame spread      : {native['mean_within_frame_std_ms']:.3f} m/s")
    print(f"  Native velocity range           : {native['minimum_ms']:.3f} to {native['maximum_ms']:.3f} m/s")

    ego = summary["native_vs_ego_only"]
    print("  Native vs ego-only Doppler")
    print(f"    Samples / bias / MAE / RMSE   : {int(ego['samples'])} / {ego.get('bias_ms', float('nan')):.3f} / {ego.get('mae_ms', float('nan')):.3f} / {ego.get('rmse_ms', float('nan')):.3f} m/s")

    detected = summary["detected_velocity_vs_ground_truth"]
    print("  Detector velocity vs lead truth")
    print(f"    Samples / bias / MAE / RMSE   : {int(detected['samples'])} / {detected.get('bias_ms', float('nan')):.3f} / {detected.get('mae_ms', float('nan')):.3f} / {detected.get('rmse_ms', float('nan')):.3f} m/s")
    print(f"    Sign agreement / correlation  : {detected.get('sign_agreement', float('nan')):.3f} / {detected.get('correlation', float('nan')):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze non-collapsed Doppler velocity metrics.")
    parser.add_argument("--input", required=True, help="closed_loop_metrics.csv from a closed-loop run")
    parser.add_argument("--json", default=None, help="Optional output path for the JSON summary")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    summary = analyze(rows)
    print_report(summary)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, allow_nan=True)
            output_file.write("\n")


if __name__ == "__main__":
    main()
