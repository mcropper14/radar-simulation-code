"""
Create a calibration JSON for semantic_pointcloud_metrics.py from CARLA camera
image size and horizontal FOV.

By default this assumes the CARLA/RaDelft spherical point clouds use a vehicle
frame with x forward, y left, z up. The generated transform maps that to the
usual pinhole camera frame: x right, y down, z forward.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def camera_matrix_from_fov(width: int, height: int, fov_degrees: float):
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ]


def identity4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def vehicle_to_camera4():
    return [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def run(args: argparse.Namespace) -> None:
    with Image.open(args.image) as image:
        width, height = image.size

    transform = identity4() if args.identity_transform else vehicle_to_camera4()
    calibration = {
        "K": camera_matrix_from_fov(width, height, args.fov),
        "T_camera_from_reference": transform,
        "T_camera_from_detection": transform,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(calibration, f, indent=2)
        f.write("\n")

    print(f"Saved calibration to: {args.out}")
    if args.identity_transform:
        print("Wrote identity transforms.")
    else:
        print("Wrote vehicle-frame to camera-frame transforms: x_fwd,y_left,z_up -> x_right,y_down,z_fwd.")
    print("Edit the transform matrices if your point clouds use a different convention.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create CARLA camera calibration JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Any camera_rgb image from the dataset.")
    parser.add_argument("--fov", type=float, default=90.0, help="CARLA camera horizontal FOV in degrees.")
    parser.add_argument("--out", type=Path, default=Path("carla_camera_calibration.json"))
    parser.add_argument(
        "--identity-transform",
        action="store_true",
        help="Use identity transforms when point clouds are already in camera coordinates.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
