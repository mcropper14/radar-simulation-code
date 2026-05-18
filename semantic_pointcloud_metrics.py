"""
Project 2D semantic segmentations onto 3D point clouds and compute
per-object detection / false-alarm metrics.

This script is intentionally segmentation-model agnostic: run a segmentation
stage first, save class-id masks as PNGs, then use this file to label point
clouds by projecting points into the aligned image plane.

Expected point cloud formats:
    cartesian: .npy array with XYZ in columns 0:3. Extra columns are preserved.
    spherical: .npy array with [range, azimuth(rad), elevation(rad), ...].

Expected calibration JSON:
    {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "T_camera_from_reference": [[... 4x4 ...]],
      "T_camera_from_detection": [[... 4x4 ...]]
    }

If both point clouds are already in the camera frame, use identity transforms.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from sklearn.neighbors import KDTree
except Exception:  # pragma: no cover - fallback for minimal installs
    KDTree = None


CLASS_NAME_TO_ID = {
    "unlabeled": 0,
    "road": 1,
    "sidewalk": 2,
    "building": 3,
    "wall": 4,
    "fence": 5,
    "pole": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "vegetation": 9,
    "terrain": 10,
    "sky": 11,
    "person": 12,
    "rider": 13,
    "car": 14,
    "truck": 15,
    "bus": 16,
    "train": 17,
    "motorcycle": 18,
    "bicycle": 19,
}

ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

DEFAULT_OBJECT_CLASS_IDS = [
    CLASS_NAME_TO_ID["car"],
    CLASS_NAME_TO_ID["person"],
    CLASS_NAME_TO_ID["truck"],
    CLASS_NAME_TO_ID["bus"],
    CLASS_NAME_TO_ID["rider"],
    CLASS_NAME_TO_ID["bicycle"],
    CLASS_NAME_TO_ID["motorcycle"],
]

GROUP_NAME_TO_CLASS_IDS = {
    "two-wheeler": (
        CLASS_NAME_TO_ID["bicycle"],
        CLASS_NAME_TO_ID["motorcycle"],
    ),
    "bike_motorcycle": (
        CLASS_NAME_TO_ID["bicycle"],
        CLASS_NAME_TO_ID["motorcycle"],
    ),
    "bicycle_motorcycle": (
        CLASS_NAME_TO_ID["bicycle"],
        CLASS_NAME_TO_ID["motorcycle"],
    ),
}

GROUP_NAME_TO_ID = {
    "two-wheeler": 201,
    "bike_motorcycle": 201,
    "bicycle_motorcycle": 201,
}


@dataclass
class Calibration:
    K: np.ndarray
    T_camera_from_reference: np.ndarray
    T_camera_from_detection: np.ndarray


@dataclass
class LabeledCloud:
    path: Path
    points: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    valid_projection: np.ndarray
    class_ids: np.ndarray


@dataclass(frozen=True)
class EvaluationClass:
    class_id: int
    class_name: str
    source_class_ids: Tuple[int, ...]


def parse_timestamp(path: Path) -> float:
    return float(path.stem)


def list_npy_files(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.glob("*.npy") if p.is_file() and not p.name.startswith("._")],
        key=parse_timestamp,
    )


def list_mask_files(folder: Path) -> List[Path]:
    files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    return sorted(files, key=parse_timestamp)


def load_calibration(path: Path) -> Calibration:
    with path.open("r") as f:
        data = json.load(f)

    K = np.asarray(data["K"], dtype=np.float64)
    T_ref = np.asarray(
        data.get("T_camera_from_reference", np.eye(4)),
        dtype=np.float64,
    )
    T_det = np.asarray(
        data.get("T_camera_from_detection", np.eye(4)),
        dtype=np.float64,
    )

    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    if T_ref.shape != (4, 4):
        raise ValueError(f"T_camera_from_reference must be 4x4, got {T_ref.shape}")
    if T_det.shape != (4, 4):
        raise ValueError(f"T_camera_from_detection must be 4x4, got {T_det.shape}")

    return Calibration(K=K, T_camera_from_reference=T_ref, T_camera_from_detection=T_det)


def spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    range_ = points[:, 0]
    azimuth = points[:, 1]
    elevation = points[:, 2]

    xyz = np.empty((points.shape[0], 3), dtype=np.float64)
    xyz[:, 0] = range_ * np.cos(elevation) * np.cos(azimuth)
    xyz[:, 1] = range_ * np.cos(elevation) * np.sin(azimuth)
    xyz[:, 2] = range_ * np.sin(elevation)
    return xyz


def load_points(path: Path, point_format: str = "cartesian", flip_y: bool = False) -> np.ndarray:
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} must be a 2D .npy array with at least XYZ columns")

    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    points = points.astype(np.float64, copy=False)

    if point_format == "cartesian":
        pass
    elif point_format == "spherical":
        xyz = spherical_to_cartesian(points)
        points = np.column_stack([xyz, points[:, 3:]]) if points.shape[1] > 3 else xyz
    else:
        raise ValueError(f"Unknown point format '{point_format}'")

    if flip_y:
        points = points.copy()
        points[:, 1] *= -1.0

    return points


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32, copy=False)


def transform_points(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.ones((xyz.shape[0], 4), dtype=np.float64)
    homog[:, :3] = xyz
    transformed = (transform @ homog.T).T
    return transformed[:, :3]


def project_points(
    xyz: np.ndarray,
    K: np.ndarray,
    T_camera_from_points: np.ndarray,
    image_shape: Tuple[int, int],
    min_depth: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_cam = transform_points(xyz, T_camera_from_points)
    depth = xyz_cam[:, 2]

    uvw = (K @ xyz_cam.T).T
    uv_float = np.full((xyz.shape[0], 2), np.nan, dtype=np.float64)
    depth_ok = depth > min_depth
    uv_float[depth_ok, 0] = uvw[depth_ok, 0] / depth[depth_ok]
    uv_float[depth_ok, 1] = uvw[depth_ok, 1] / depth[depth_ok]

    h, w = image_shape
    finite_uv = np.isfinite(uv_float).all(axis=1)
    valid = (
        depth_ok
        & finite_uv
        & (uv_float[:, 0] >= 0)
        & (uv_float[:, 0] < w)
        & (uv_float[:, 1] >= 0)
        & (uv_float[:, 1] < h)
    )

    uv = np.zeros((xyz.shape[0], 2), dtype=np.int32)
    uv[valid, 0] = np.rint(uv_float[valid, 0]).astype(np.int32)
    uv[valid, 1] = np.rint(uv_float[valid, 1]).astype(np.int32)
    uv[valid, 0] = np.clip(uv[valid, 0], 0, w - 1)
    uv[valid, 1] = np.clip(uv[valid, 1], 0, h - 1)
    return uv, depth, valid


def label_points_from_mask(
    cloud_path: Path,
    mask: np.ndarray,
    K: np.ndarray,
    T_camera_from_points: np.ndarray,
    point_format: str = "cartesian",
    flip_y: bool = False,
    min_depth: float = 1e-6,
) -> LabeledCloud:
    points = load_points(cloud_path, point_format=point_format, flip_y=flip_y)
    uv, depth, valid = project_points(
        points[:, :3],
        K=K,
        T_camera_from_points=T_camera_from_points,
        image_shape=mask.shape[:2],
        min_depth=min_depth,
    )

    class_ids = np.zeros(points.shape[0], dtype=np.int32)
    class_ids[valid] = mask[uv[valid, 1], uv[valid, 0]]

    return LabeledCloud(
        path=cloud_path,
        points=points,
        uv=uv,
        depth=depth,
        valid_projection=valid,
        class_ids=class_ids,
    )


def find_closest_file(
    target_ts: float,
    candidates: Sequence[Path],
    candidate_ts: np.ndarray,
) -> Tuple[Optional[Path], float]:
    idx = np.searchsorted(candidate_ts, target_ts)
    best_file = None
    best_gap = float("inf")

    for j in (idx - 1, idx):
        if 0 <= j < len(candidates):
            gap = abs(candidate_ts[j] - target_ts)
            if gap < best_gap:
                best_gap = gap
                best_file = candidates[j]

    return best_file, best_gap


def parse_evaluation_classes(
    values: Optional[Sequence[str]],
    merge_two_wheelers: bool = False,
) -> List[EvaluationClass]:
    raw_values = list(values) if values else [
        "car",
        "person",
        "truck",
        "bus",
        "rider",
        "bicycle",
        "motorcycle",
    ]

    if merge_two_wheelers:
        raw_values = [
            value for value in raw_values
            if value.strip().lower() not in {"bicycle", "19", "motorcycle", "18"}
        ]
        raw_values.append("two-wheeler")

    eval_classes = []
    seen = set()

    for value in raw_values:
        value = value.strip().lower()

        if value in GROUP_NAME_TO_CLASS_IDS:
            class_id = GROUP_NAME_TO_ID[value]
            class_name = "two-wheeler"
            source_class_ids = GROUP_NAME_TO_CLASS_IDS[value]
        elif value.isdigit():
            class_id = int(value)
            class_name = ID_TO_CLASS_NAME.get(class_id, f"class_{class_id}")
            source_class_ids = (class_id,)
        else:
            if value not in CLASS_NAME_TO_ID:
                known = sorted(list(CLASS_NAME_TO_ID) + list(GROUP_NAME_TO_CLASS_IDS))
                raise ValueError(f"Unknown class or group '{value}'. Known: {known}")
            class_id = CLASS_NAME_TO_ID[value]
            class_name = value
            source_class_ids = (class_id,)

        key = (class_name, source_class_ids)
        if key in seen:
            continue
        seen.add(key)
        eval_classes.append(
            EvaluationClass(
                class_id=class_id,
                class_name=class_name,
                source_class_ids=tuple(source_class_ids),
            )
        )

    return eval_classes


def nearest_neighbor_distances(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    if source_xyz.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if target_xyz.shape[0] == 0:
        return np.full(source_xyz.shape[0], np.inf, dtype=np.float64)

    if KDTree is not None:
        tree = KDTree(target_xyz)
        dists, _ = tree.query(source_xyz, k=1)
        return dists[:, 0]

    diff = source_xyz[:, None, :] - target_xyz[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def count_reference_matches(
    reference_xyz: np.ndarray,
    detection_xyz: np.ndarray,
    match_distance: float,
) -> int:
    if reference_xyz.shape[0] == 0 or detection_xyz.shape[0] == 0:
        return 0
    dists = nearest_neighbor_distances(reference_xyz, detection_xyz)
    return int(np.count_nonzero(dists <= match_distance))


def count_false_alarm_points(
    detection_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    match_distance: float,
) -> int:
    if detection_xyz.shape[0] == 0:
        return 0
    dists = nearest_neighbor_distances(detection_xyz, reference_xyz)
    return int(np.count_nonzero(dists > match_distance))


def compute_detection_rows(
    reference: LabeledCloud,
    detection: LabeledCloud,
    eval_classes: Sequence[EvaluationClass],
    match_distance: float,
) -> List[Dict[str, object]]:
    rows = []

    for eval_class in eval_classes:
        ref_mask = reference.valid_projection & np.isin(reference.class_ids, eval_class.source_class_ids)
        det_mask = detection.valid_projection & np.isin(detection.class_ids, eval_class.source_class_ids)

        ref_xyz = reference.points[ref_mask, :3]
        det_xyz = detection.points[det_mask, :3]

        true_positive_ref_points = count_reference_matches(ref_xyz, det_xyz, match_distance)
        false_negative_points = int(ref_xyz.shape[0] - true_positive_ref_points)
        false_alarm_points = count_false_alarm_points(det_xyz, ref_xyz, match_distance)

        reference_points = int(ref_xyz.shape[0])
        detection_points = int(det_xyz.shape[0])
        p_detection = true_positive_ref_points / reference_points if reference_points else np.nan
        p_false_alarm = false_alarm_points / detection_points if detection_points else np.nan

        rows.append(
            {
                "class_id": int(eval_class.class_id),
                "class_name": eval_class.class_name,
                "source_class_ids": "|".join(str(cid) for cid in eval_class.source_class_ids),
                "source_class_names": "|".join(
                    ID_TO_CLASS_NAME.get(cid, f"class_{cid}") for cid in eval_class.source_class_ids
                ),
                "reference_points": reference_points,
                "detection_points": detection_points,
                "true_positive_ref_points": true_positive_ref_points,
                "false_negative_points": false_negative_points,
                "false_alarm_points": false_alarm_points,
                "p_detection": p_detection,
                "p_false_alarm": p_false_alarm,
            }
        )

    return rows


def save_labeled_cloud(
    labeled: LabeledCloud,
    eval_classes: Sequence[EvaluationClass],
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_ids = sorted({cid for eval_class in eval_classes for cid in eval_class.source_class_ids})
    keep = labeled.valid_projection & np.isin(labeled.class_ids, class_ids)
    labeled_points = np.column_stack([labeled.points[keep], labeled.class_ids[keep]])
    np.save(out_dir / f"{labeled.path.stem}_{prefix}_semantic.npy", labeled_points)


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(
    rows: Sequence[Dict[str, object]],
    eval_classes: Sequence[EvaluationClass],
) -> List[Dict[str, object]]:
    summary = []
    for eval_class in eval_classes:
        class_rows = [r for r in rows if int(r["class_id"]) == int(eval_class.class_id)]
        ref = sum(int(r["reference_points"]) for r in class_rows)
        det = sum(int(r["detection_points"]) for r in class_rows)
        tp = sum(int(r["true_positive_ref_points"]) for r in class_rows)
        fn = sum(int(r["false_negative_points"]) for r in class_rows)
        fa = sum(int(r["false_alarm_points"]) for r in class_rows)

        summary.append(
            {
                "class_id": int(eval_class.class_id),
                "class_name": eval_class.class_name,
                "source_class_ids": "|".join(str(cid) for cid in eval_class.source_class_ids),
                "source_class_names": "|".join(
                    ID_TO_CLASS_NAME.get(cid, f"class_{cid}") for cid in eval_class.source_class_ids
                ),
                "reference_points": ref,
                "detection_points": det,
                "true_positive_ref_points": tp,
                "false_negative_points": fn,
                "false_alarm_points": fa,
                "p_detection": tp / ref if ref else np.nan,
                "p_false_alarm": fa / det if det else np.nan,
            }
        )
    return summary


def run(args: argparse.Namespace) -> None:
    calibration = load_calibration(args.calibration)
    eval_classes = parse_evaluation_classes(
        args.classes,
        merge_two_wheelers=args.merge_two_wheelers,
    )

    reference_files = list_npy_files(args.reference_cloud_dir)
    detection_files = list_npy_files(args.detection_cloud_dir)
    mask_files = list_mask_files(args.mask_dir)

    if not reference_files:
        raise RuntimeError(f"No .npy files found in {args.reference_cloud_dir}")
    if not detection_files:
        raise RuntimeError(f"No .npy files found in {args.detection_cloud_dir}")
    if not mask_files:
        raise RuntimeError(f"No mask image files found in {args.mask_dir}")

    detection_ts = np.array([parse_timestamp(p) for p in detection_files], dtype=np.float64)
    mask_ts = np.array([parse_timestamp(p) for p in mask_files], dtype=np.float64)

    per_frame_rows = []
    skipped = 0

    for reference_path in reference_files:
        ref_ts = parse_timestamp(reference_path)
        detection_path, detection_gap = find_closest_file(ref_ts, detection_files, detection_ts)
        mask_path, mask_gap = find_closest_file(ref_ts, mask_files, mask_ts)

        if detection_path is None or detection_gap > args.max_cloud_time_diff:
            skipped += 1
            print(f"SKIP {reference_path.name}: no detection cloud within time threshold")
            continue

        if mask_path is None or mask_gap > args.max_mask_time_diff:
            skipped += 1
            print(f"SKIP {reference_path.name}: no mask within time threshold")
            continue

        mask = load_mask(mask_path)
        reference = label_points_from_mask(
            reference_path,
            mask,
            calibration.K,
            calibration.T_camera_from_reference,
            point_format=args.reference_format,
            flip_y=args.flip_reference_y,
            min_depth=args.min_depth,
        )
        detection = label_points_from_mask(
            detection_path,
            mask,
            calibration.K,
            calibration.T_camera_from_detection,
            point_format=args.detection_format,
            flip_y=args.flip_detection_y,
            min_depth=args.min_depth,
        )

        frame_rows = compute_detection_rows(
            reference=reference,
            detection=detection,
            eval_classes=eval_classes,
            match_distance=args.match_distance,
        )

        for row in frame_rows:
            row.update(
                {
                    "reference_file": reference_path.name,
                    "detection_file": detection_path.name,
                    "mask_file": mask_path.name,
                    "reference_ts": ref_ts,
                    "detection_ts": parse_timestamp(detection_path),
                    "mask_ts": parse_timestamp(mask_path),
                    "cloud_time_gap_s": detection_gap,
                    "mask_time_gap_s": mask_gap,
                }
            )
        per_frame_rows.extend(frame_rows)

        if args.save_filtered_clouds:
            save_labeled_cloud(reference, eval_classes, args.out_dir / "filtered_clouds", "reference")
            save_labeled_cloud(detection, eval_classes, args.out_dir / "filtered_clouds", "detection")

        print(
            f"OK {reference_path.name} <-> {detection_path.name}, "
            f"mask={mask_path.name}, cloud_gap={detection_gap:.6f}s, mask_gap={mask_gap:.6f}s"
        )

    if not per_frame_rows:
        print(f"No valid matched frames. Skipped: {skipped}")
        return

    summary_rows = summarize_rows(per_frame_rows, eval_classes)
    write_rows(args.out_dir / "per_frame_detection_false_alarm.csv", per_frame_rows)
    write_rows(args.out_dir / "summary_detection_false_alarm.csv", summary_rows)

    print("\n=== Summary ===")
    print(f"Frames processed: {len({r['reference_file'] for r in per_frame_rows})}")
    print(f"Skipped:          {skipped}")
    for row in summary_rows:
        pdet = row["p_detection"]
        pfa = row["p_false_alarm"]
        print(
            f"{row['class_name']:>12s}: "
            f"P(det)={pdet:.4f}  P(false alarm)={pfa:.4f}  "
            f"ref={row['reference_points']} det={row['detection_points']}"
        )
    print(f"\nSaved outputs to: {args.out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project semantic masks onto point clouds and compute P(detection)/P(false alarm)."
    )
    parser.add_argument("--reference-cloud-dir", type=Path, required=True)
    parser.add_argument("--detection-cloud-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("semantic_pointcloud_metrics_out"))
    parser.add_argument("--classes", nargs="*", default=None, help="Class names or IDs. Defaults to object classes.")
    parser.add_argument(
        "--merge-two-wheelers",
        action="store_true",
        help="Evaluate bicycle and motorcycle together as one two-wheeler class.",
    )
    parser.add_argument("--reference-format", choices=["cartesian", "spherical"], default="cartesian")
    parser.add_argument("--detection-format", choices=["cartesian", "spherical"], default="cartesian")
    parser.add_argument("--match-distance", type=float, default=0.5, help="3D match radius in point-cloud units.")
    parser.add_argument("--max-cloud-time-diff", type=float, default=0.02)
    parser.add_argument("--max-mask-time-diff", type=float, default=0.02)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--flip-reference-y", action="store_true")
    parser.add_argument("--flip-detection-y", action="store_true")
    parser.add_argument("--save-filtered-clouds", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
