import os
from glob import glob
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from transformers import pipeline



IMG_DIR = "/home/cropthecoder/Downloads/Scene1_rosDS/Scene1/rosDS/ueye_left_image_rect_color"
OUT_DIR = "./radelft_segmented"

RAW_MASK_DIR = os.path.join(OUT_DIR, "mask_raw")
COLOR_MASK_DIR = os.path.join(OUT_DIR, "mask_color")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlay")
HIST_DIR = os.path.join(OUT_DIR, "hist")

os.makedirs(RAW_MASK_DIR, exist_ok=True)
os.makedirs(COLOR_MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)


MODEL_NAME = "facebook/mask2former-swin-large-cityscapes-panoptic"


segmenter = pipeline("image-segmentation", model=MODEL_NAME)


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


CITYSCAPES_TO_TARGET = {
    "road": CLASS_NAME_TO_ID["road"],
    "sidewalk": CLASS_NAME_TO_ID["sidewalk"],
    "building": CLASS_NAME_TO_ID["building"],
    "wall": CLASS_NAME_TO_ID["wall"],
    "fence": CLASS_NAME_TO_ID["fence"],
    "pole": CLASS_NAME_TO_ID["pole"],
    "traffic light": CLASS_NAME_TO_ID["traffic light"],
    "traffic sign": CLASS_NAME_TO_ID["traffic sign"],
    "vegetation": CLASS_NAME_TO_ID["vegetation"],
    "terrain": CLASS_NAME_TO_ID["terrain"],
    "sky": CLASS_NAME_TO_ID["sky"],
    "person": CLASS_NAME_TO_ID["person"],
    "rider": CLASS_NAME_TO_ID["rider"],
    "car": CLASS_NAME_TO_ID["car"],
    "truck": CLASS_NAME_TO_ID["truck"],
    "bus": CLASS_NAME_TO_ID["bus"],
    "train": CLASS_NAME_TO_ID["train"],
    "motorcycle": CLASS_NAME_TO_ID["motorcycle"],
    "bicycle": CLASS_NAME_TO_ID["bicycle"],
}


CLASS_COLORS = {
    0: (0, 0, 0),          # unlabeled
    1: (128, 64, 128),     # road
    2: (244, 35, 232),     # sidewalk
    3: (70, 70, 70),       # building
    4: (102, 102, 156),    # wall
    5: (190, 153, 153),    # fence
    6: (153, 153, 153),    # pole
    7: (250, 170, 30),     # traffic light
    8: (220, 220, 0),      # traffic sign
    9: (107, 142, 35),     # vegetation
    10: (152, 251, 152),   # terrain
    11: (70, 130, 180),    # sky
    12: (220, 20, 60),     # person
    13: (255, 0, 0),       # rider
    14: (0, 0, 142),       # car
    15: (0, 0, 70),        # truck
    16: (0, 60, 100),      # bus
    17: (0, 80, 100),      # train
    18: (0, 0, 230),       # motorcycle
    19: (119, 11, 32),     # bicycle
}


def make_color_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def make_overlay(rgb_image, color_mask, alpha=0.5):
    # rgb_image expected RGB uint8
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, color_mask[:, :, ::-1], alpha, 0)
    # color_mask is BGR, convert to RGB before blending
    return overlay


def segment_and_remap(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    results = segmenter(image_pil)

    out_mask = np.zeros((h, w), dtype=np.uint8)

    for seg in results:
        label = seg["label"].lower().strip()

        if label not in CITYSCAPES_TO_TARGET:
            continue

        class_id = CITYSCAPES_TO_TARGET[label]

        mask = np.array(seg["mask"])
        if mask.ndim == 3:
            mask = mask[..., 0]

        # some outputs may be 0/255 or boolean-like
        binary_mask = mask > 0
        out_mask[binary_mask] = class_id

    return image_np, out_mask


def save_outputs(image_path, image_np, mask):
    base = os.path.splitext(os.path.basename(image_path))[0]

    raw_mask_path = os.path.join(RAW_MASK_DIR, f"{base}.png")
    color_mask_path = os.path.join(COLOR_MASK_DIR, f"{base}.png")
    overlay_path = os.path.join(OVERLAY_DIR, f"{base}.png")

    color_mask_bgr = make_color_mask(mask)
    color_mask_rgb = color_mask_bgr[:, :, ::-1]
    overlay = ((0.5 * image_np) + (0.5 * color_mask_rgb)).astype(np.uint8)

    Image.fromarray(mask).save(raw_mask_path)
    Image.fromarray(color_mask_rgb).save(color_mask_path)
    Image.fromarray(overlay).save(overlay_path)


def accumulate_histogram(mask, hist_counts):
    unique_ids, counts = np.unique(mask, return_counts=True)
    for cid, count in zip(unique_ids, counts):
        hist_counts[int(cid)] += int(count)


def save_histogram(hist_counts):
    class_ids = sorted(ID_TO_CLASS_NAME.keys())
    class_names = [ID_TO_CLASS_NAME[cid] for cid in class_ids]
    counts = [hist_counts.get(cid, 0) for cid in class_ids]

    # save csv
    csv_path = os.path.join(HIST_DIR, "class_histogram.csv")
    with open(csv_path, "w") as f:
        f.write("class_id,class_name,pixel_count\n")
        for cid, name, count in zip(class_ids, class_names, counts):
            f.write(f"{cid},{name},{count}\n")

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Pixel Count")
    plt.title("Carla Segmentation Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(HIST_DIR, "class_histogram.png"), dpi=200)
    plt.close()


def print_mask_summary(mask, image_path):
    unique_ids = np.unique(mask)
    class_names = [ID_TO_CLASS_NAME[int(cid)] for cid in unique_ids]
    print(f"\n{os.path.basename(image_path)}")
    print("Unique class IDs:", unique_ids.tolist())
    print("Unique class names:", class_names)


def main():
    image_paths = sorted(glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")))
    image_paths = image_paths[:3000] #have 3000 carla images

    if len(image_paths) == 0:
        print(f"No images found in {IMG_DIR}")
        return

    hist_counts = defaultdict(int)

    for idx, image_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] Processing {image_path}")

        image_np, mask = segment_and_remap(image_path)
        save_outputs(image_path, image_np, mask)
        accumulate_histogram(mask, hist_counts)

        # print class summary for first few images so you can verify
        if idx < 10:
            print_mask_summary(mask, image_path)

    save_histogram(hist_counts)
    print("\nDone.")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()