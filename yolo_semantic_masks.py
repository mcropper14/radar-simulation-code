"""
Run YOLO segmentation on CARLA/RaDelft camera frames and save class-id masks.

This is the missing stage before semantic point-cloud filtering:

    radelf_spherical/
        camera_rgb/*.png or *.jpg
        lidar_spher/*.npy
        radar_spher/*.npy

becomes:

    yolo_segmented/
        mask_raw/*.png      # one-channel class IDs matching segment3.py
        mask_color/*.png
        overlay/*.png
        debug/*_segments.txt
        hist/*.csv

The script uses Ultralytics YOLO segmentation models. Pass your YOLO26 model
checkpoint with --model if you have one; otherwise any Ultralytics segmentation
checkpoint path/name that exposes masks will work.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


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

CLASS_COLORS_RGB = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32),
}

# COCO/Ultralytics names -> local Cityscapes-style IDs used elsewhere here.
YOLO_TO_TARGET = {
    "person": CLASS_NAME_TO_ID["person"],
    "rider": CLASS_NAME_TO_ID["rider"],
    "bicycle": CLASS_NAME_TO_ID["bicycle"],
    "car": CLASS_NAME_TO_ID["car"],
    "motorcycle": CLASS_NAME_TO_ID["motorcycle"],
    "motorbike": CLASS_NAME_TO_ID["motorcycle"],
    "bus": CLASS_NAME_TO_ID["bus"],
    "truck": CLASS_NAME_TO_ID["truck"],
    "train": CLASS_NAME_TO_ID["train"],
    "traffic light": CLASS_NAME_TO_ID["traffic light"],
    "traffic sign": CLASS_NAME_TO_ID["traffic sign"],
    "stop sign": CLASS_NAME_TO_ID["traffic sign"],
}

THING_CLASS_IDS = [
    CLASS_NAME_TO_ID["car"],
    CLASS_NAME_TO_ID["person"],
    CLASS_NAME_TO_ID["truck"],
    CLASS_NAME_TO_ID["bus"],
    CLASS_NAME_TO_ID["rider"],
    CLASS_NAME_TO_ID["bicycle"],
    CLASS_NAME_TO_ID["motorcycle"],
]


def get_image_paths(image_dir: Path) -> List[Path]:
    paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(image_dir.glob(suffix))
    return sorted(paths, key=lambda p: float(p.stem))


def make_color_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in CLASS_COLORS_RGB.items():
        color[mask == class_id] = rgb
    return color


def make_overlay(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * image_rgb + alpha * color_mask).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image = image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def tensor_to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def segment_image(model, image_path: Path, conf: float) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image_rgb.shape[:2]

    results = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False,
        retina_masks=True,
    )
    result = results[0]

    class_mask = np.zeros((height, width), dtype=np.uint8)
    segment_summary = []

    if result.masks is None or result.boxes is None:
        return class_mask, segment_summary

    masks = tensor_to_numpy(result.masks.data)
    class_indices = tensor_to_numpy(result.boxes.cls).astype(int)
    scores = tensor_to_numpy(result.boxes.conf).astype(float)
    names = result.names

    # Write lower-confidence masks first, so stronger detections win overlaps.
    order = np.argsort(scores)
    for out_idx, det_idx in enumerate(order, start=1):
        yolo_class_id = int(class_indices[det_idx])
        yolo_name = str(names[yolo_class_id]).lower().strip()
        target_class_id = YOLO_TO_TARGET.get(yolo_name)
        if target_class_id is None:
            continue

        binary = masks[det_idx] > 0.5
        if binary.shape != (height, width):
            binary = resize_binary_mask(binary, (width, height))

        class_mask[binary] = target_class_id
        segment_summary.append(
            {
                "segment_id": out_idx,
                "label_id": yolo_class_id,
                "class_name": yolo_name,
                "target_class_id": target_class_id,
                "pixels": int(np.count_nonzero(binary)),
                "score": float(scores[det_idx]),
            }
        )

    return class_mask, segment_summary


def save_debug(path: Path, image_path: Path, segment_summary: Sequence[Dict[str, object]]) -> None:
    with path.open("w") as f:
        f.write(f"Image: {image_path}\n")
        f.write("segment_id,label_id,class_name,target_class_id,pixels,score\n")
        for row in segment_summary:
            f.write(
                f"{row['segment_id']},{row['label_id']},{row['class_name']},"
                f"{row['target_class_id']},{row['pixels']},{row['score']:.6f}\n"
            )


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_histograms(
    hist_dir: Path,
    total_pixel_counts: Dict[int, int],
    per_frame_instance_counts: Dict[int, List[int]],
    instance_pixel_sums: Dict[int, int],
    instance_counts: Dict[int, int],
    num_processed: int,
    num_failed: int,
    num_requested: int,
) -> None:
    all_class_ids = sorted(ID_TO_CLASS_NAME)

    write_csv(
        hist_dir / "total_pixel_histogram.csv",
        ["class_id", "class_name", "total_pixels"],
        [(cid, ID_TO_CLASS_NAME[cid], total_pixel_counts.get(cid, 0)) for cid in all_class_ids],
    )

    write_csv(
        hist_dir / "avg_instances_per_frame_thing_classes.csv",
        ["class_id", "class_name", "avg_instances_per_frame"],
        [
            (
                cid,
                ID_TO_CLASS_NAME[cid],
                float(np.mean(per_frame_instance_counts[cid])) if per_frame_instance_counts[cid] else 0.0,
            )
            for cid in THING_CLASS_IDS
        ],
    )

    write_csv(
        hist_dir / "avg_pixels_per_instance_thing_classes.csv",
        ["class_id", "class_name", "avg_pixels_per_instance", "instance_count"],
        [
            (
                cid,
                ID_TO_CLASS_NAME[cid],
                float(instance_pixel_sums[cid] / instance_counts[cid]) if instance_counts[cid] else 0.0,
                instance_counts[cid],
            )
            for cid in THING_CLASS_IDS
        ],
    )

    write_csv(
        hist_dir / "run_summary.csv",
        ["num_requested", "num_processed", "num_failed"],
        [(num_requested, num_processed, num_failed)],
    )


def run(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLO segmentation. Install it with: "
            "pip install ultralytics"
        ) from exc

    image_paths = get_image_paths(args.image_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    raw_dir = args.out_dir / "mask_raw"
    color_dir = args.out_dir / "mask_color"
    overlay_dir = args.out_dir / "overlay"
    debug_dir = args.out_dir / "debug"
    hist_dir = args.out_dir / "hist"
    for directory in (raw_dir, color_dir, overlay_dir, debug_dir, hist_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))

    total_pixel_counts = defaultdict(int)
    per_frame_instance_counts = defaultdict(list)
    instance_pixel_sums = defaultdict(int)
    instance_counts = defaultdict(int)

    num_failed = 0
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path.name}")

        try:
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            class_mask, segment_summary = segment_image(model, image_path, args.conf)
        except Exception as exc:
            print(f"  Failed: {exc}")
            num_failed += 1
            continue

        base = image_path.stem
        color_mask = make_color_mask(class_mask)
        overlay = make_overlay(image_rgb, color_mask, args.alpha)

        Image.fromarray(class_mask).save(raw_dir / f"{base}.png")
        Image.fromarray(color_mask).save(color_dir / f"{base}.png")
        Image.fromarray(overlay).save(overlay_dir / f"{base}.png")
        save_debug(debug_dir / f"{base}_segments.txt", image_path, segment_summary)

        unique_ids, counts = np.unique(class_mask, return_counts=True)
        for class_id, count in zip(unique_ids, counts):
            total_pixel_counts[int(class_id)] += int(count)

        frame_instance_map = defaultdict(int)
        for row in segment_summary:
            cid = int(row["target_class_id"])
            frame_instance_map[cid] += 1
            if cid in THING_CLASS_IDS:
                instance_counts[cid] += 1
                instance_pixel_sums[cid] += int(row["pixels"])

        for cid in THING_CLASS_IDS:
            per_frame_instance_counts[cid].append(frame_instance_map.get(cid, 0))

    num_processed = len(image_paths) - num_failed
    save_histograms(
        hist_dir=hist_dir,
        total_pixel_counts=total_pixel_counts,
        per_frame_instance_counts=per_frame_instance_counts,
        instance_pixel_sums=instance_pixel_sums,
        instance_counts=instance_counts,
        num_processed=num_processed,
        num_failed=num_failed,
        num_requested=len(image_paths),
    )

    print("\nDone.")
    print(f"Processed: {num_processed}")
    print(f"Failed:    {num_failed}")
    print(f"Saved outputs to: {args.out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation and save class-id masks.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("yolo_segmented"))
    parser.add_argument("--model", type=Path, default=Path("yolo11x-seg.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
