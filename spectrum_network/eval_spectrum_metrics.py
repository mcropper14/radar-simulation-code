import os
import glob
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# -------------------------
# Chamfer helpers
# -------------------------
def ucd(source: np.ndarray, target: np.ndarray) -> float:
    """Unidirectional chamfer: mean NN distance from source -> target."""
    if source.size == 0 or target.size == 0:
        return float("inf")
    tree = cKDTree(target)
    d, _ = tree.query(source, k=1)
    return float(np.mean(d))

def bcd(source: np.ndarray, target: np.ndarray) -> float:
    """Bidirectional chamfer: ucd(source,target) + ucd(target,source)."""
    return ucd(source, target) + ucd(target, source)


# -------------------------
# Dataset
# -------------------------
class SpectrumPairDataset(Dataset):
    def __init__(self, radar_dir, target_dir, transform=None, limit=None):
        radar_files = sorted(glob.glob(os.path.join(radar_dir, "*")))
        target_files = sorted(glob.glob(os.path.join(target_dir, "*")))

        radar_base = {os.path.basename(f): f for f in radar_files}
        target_base = {os.path.basename(f): f for f in target_files}
        common = sorted(set(radar_base.keys()) & set(target_base.keys()))

        if limit is not None:
            common = common[:limit]

        self.items = [(radar_base[n], target_base[n], n) for n in common]
        self.transform = transform

        if len(self.items) == 0:
            raise RuntimeError("No matching files found between radar_dir and target_dir.")
        print(f"[SpectrumPairDataset] pairs={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        radar_path, target_path, name = self.items[idx]
        radar_img = Image.open(radar_path).convert("L")
        target_img = Image.open(target_path).convert("L")

        if self.transform:
            radar_t = self.transform(radar_img)
            target_t = self.transform(target_img)
        else:
            radar_t = transforms.ToTensor()(radar_img)
            target_t = transforms.ToTensor()(target_img)

        return radar_t, target_t, name


# -------------------------
# Model (MUST match training)
# -------------------------
def make_resnet101_regressor():
    # Use weights=None so your checkpoint keys match exactly (no pretrained head assumptions).
    model = models.resnet101(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 256 * 256)
    return model


# -------------------------
# Point extraction from image
# -------------------------
def map_to_points(
    img_2d: np.ndarray,
    thr: float,
    max_points: int,
    normalize: str,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Extract points where img >= thr.
    Returns Nx2 points in:
      - unit:   [0,1]x[0,1]
      - signed: [-1,1]x[-1,1]
      - pixel:  pixel coords
    """
    H, W = img_2d.shape
    ys, xs = np.where(img_2d >= thr)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    if normalize == "unit":
        pts[:, 0] /= max(W - 1, 1)
        pts[:, 1] /= max(H - 1, 1)
    elif normalize == "signed":
        pts[:, 0] = (pts[:, 0] / max(W - 1, 1)) * 2.0 - 1.0
        pts[:, 1] = (pts[:, 1] / max(H - 1, 1)) * 2.0 - 1.0
    elif normalize == "pixel":
        pass
    else:
        raise ValueError(f"Unknown normalize='{normalize}'")

    return pts


# -------------------------
# Failure visualization
# -------------------------
def save_failure_viz(out_dir, stem, pred01, gt01, thr_pred, thr_gt):
    """
    pred01/gt01 are 0..1 float images (H,W).
    Saves:
      - FP mask (pred=1, gt=0)
      - FN mask (pred=0, gt=1)
      - overlay view
    """
    os.makedirs(out_dir, exist_ok=True)

    pr = (pred01 >= thr_pred).astype(np.uint8)
    gt = (gt01 >= thr_gt).astype(np.uint8)

    fp = np.logical_and(pr == 1, gt == 0).astype(np.uint8)
    fn = np.logical_and(pr == 0, gt == 1).astype(np.uint8)

    # Save binary masks
    Image.fromarray((fp * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{stem}_fp.png"))
    Image.fromarray((fn * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{stem}_fn.png"))

    # Overlay (gt in gray, fp in red-ish, fn in blue-ish)
    base = (np.clip(gt01, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)

    # FP -> brighten red channel
    rgb[..., 0] = np.clip(rgb[..., 0] + fp * 180, 0, 255)
    # FN -> brighten blue channel
    rgb[..., 2] = np.clip(rgb[..., 2] + fn * 180, 0, 255)

    Image.fromarray(rgb).save(os.path.join(out_dir, f"{stem}_overlay.png"))

    # Simple “signature” for similarity checks
    H, W = fp.shape
    top = slice(0, H // 3)
    mid = slice(H // 3, 2 * H // 3)
    bot = slice(2 * H // 3, H)

    sig = {
        "fp_rate": float(fp.mean()),
        "fn_rate": float(fn.mean()),
        "fp_top": float(fp[top, :].mean()),
        "fp_mid": float(fp[mid, :].mean()),
        "fp_bot": float(fp[bot, :].mean()),
        "fn_top": float(fn[top, :].mean()),
        "fn_mid": float(fn[mid, :].mean()),
        "fn_bot": float(fn[bot, :].mean()),
    }
    return sig


# -------------------------
# Main eval
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_dir", default="spectrum_out/radar_depth_map_spectrum")
    ap.add_argument("--target_dir", default="spectrum_out/lidar_depth_map_spectrum")
    ap.add_argument("--ckpt", default="spectrum_out/spectrum_model_best.pt")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)

    # Binarization thresholds
    ap.add_argument("--thr_gt", type=float, default=0.30, help="GT threshold for positives in [0,1]")
    ap.add_argument("--thr_pred_mode", choices=["fixed", "quantile"], default="quantile")
    ap.add_argument("--thr_pred", type=float, default=0.50, help="used if thr_pred_mode=fixed")
    ap.add_argument("--pred_quantile", type=float, default=0.75, help="used if thr_pred_mode=quantile")

    # Chamfer point extraction
    ap.add_argument("--thr_pts", type=float, default=0.50, help="intensity threshold for point extraction")
    ap.add_argument("--max_points", type=int, default=20000)
    ap.add_argument("--normalize", default="unit", choices=["unit", "signed", "pixel"])

    # Outputs
    ap.add_argument("--out_dir", default="spectrum_eval_metrics")
    ap.add_argument("--num_failures", type=int, default=20, help="save top-N failure frames by worst BCD")
    ap.add_argument("--num_workers", type=int, default=0, help="set 0 if debugging hangs/slow I/O")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    ds = SpectrumPairDataset(args.radar_dir, args.target_dir, transform=tfm, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_resnet101_regressor().to(device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    rng = np.random.default_rng(0)

    # -------- Determine pred threshold (optional quantile) --------
    if args.thr_pred_mode == "quantile":
        all_vals = []
        with torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(device)
                pred = model(xb).view(-1, 1, 256, 256)
                pred = torch.sigmoid(pred)  # keep consistent with your ROC approach
                all_vals.append(pred.detach().cpu().numpy().reshape(-1))
        all_vals = np.concatenate(all_vals, axis=0)
        thr_pred = float(np.quantile(all_vals, args.pred_quantile))
    else:
        thr_pred = float(args.thr_pred)

    print(f"[thresholds] thr_pred={thr_pred:.4f} ({args.thr_pred_mode}) | thr_gt={args.thr_gt:.2f} | thr_pts={args.thr_pts:.2f}")

    # -------- Eval loop --------
    per = []
    mae_list, rel_list, ucd_list, bcd_list = [], [], [], []
    eps = 1e-8

    with torch.no_grad():
        for xb, yb, names in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb).view(-1, 1, 256, 256)
            pred = torch.sigmoid(pred)

            pred_np = pred.squeeze(1).cpu().numpy()  # (B,H,W)
            gt_np = yb.squeeze(1).cpu().numpy()

            for i in range(pred_np.shape[0]):
                name = str(names[i])
                stem = os.path.splitext(name)[0]

                p = pred_np[i]
                g = gt_np[i]

                mae = float(np.mean(np.abs(p - g)))
                rel = float(np.mean(np.abs(p - g) / (g + eps)))

                pts_p = map_to_points(p, thr=args.thr_pts, max_points=args.max_points, normalize=args.normalize, rng=rng)
                pts_g = map_to_points(g, thr=args.thr_pts, max_points=args.max_points, normalize=args.normalize, rng=rng)

                u = ucd(pts_p, pts_g)
                b = bcd(pts_p, pts_g)

                mae_list.append(mae)
                rel_list.append(rel)
                ucd_list.append(u)
                bcd_list.append(b)

                per.append({
                    "name": name,
                    "mae": mae,
                    "rel": rel,
                    "ucd": u,
                    "bcd": b,
                    "pred_pos_rate": float((p >= thr_pred).mean()),
                    "gt_pos_rate": float((g >= args.thr_gt).mean()),
                })

    # -------- Save metrics --------
    with open(os.path.join(args.out_dir, "per_frame_metrics.json"), "w") as f:
        json.dump(per, f, indent=2)

    def summarize(arr, finite_only=True):
        arr = np.asarray(arr, dtype=np.float64)
        if finite_only:
            arr = arr[np.isfinite(arr)]
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "std": float(arr.std()) if arr.size else None,
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
        }

    summary = {
        "mae": summarize(mae_list, finite_only=False),
        "rel": summarize(rel_list, finite_only=False),
        "ucd": summarize(ucd_list, finite_only=True),
        "bcd": summarize(bcd_list, finite_only=True),
        "thresholds": {
            "thr_pred": thr_pred,
            "thr_gt": args.thr_gt,
            "thr_pts": args.thr_pts,
            "thr_pred_mode": args.thr_pred_mode,
        }
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print("MAE:", summary["mae"])
    print("REL:", summary["rel"])
    print("UCD:", summary["ucd"])
    print("BCD:", summary["bcd"])
    print(f"[saved] {args.out_dir}/per_frame_metrics.json, {args.out_dir}/summary.json")

    # -------- Failure set: save top-N by worst BCD and compute signatures --------
    # Sort worst (largest BCD), ignoring inf
    per_sorted = sorted(
        [r for r in per if np.isfinite(r["bcd"])],
        key=lambda r: r["bcd"],
        reverse=True
    )
    worst = per_sorted[: args.num_failures]

    # Re-run those frames and save FP/FN masks + overlay + signature
    fail_dir = os.path.join(args.out_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)

    # Build a quick name->index lookup
    name_to_idx = {ds.items[i][2]: i for i in range(len(ds))}

    failure_rows = []
    for r in worst:
        name = r["name"]
        idx = name_to_idx.get(name, None)
        if idx is None:
            continue

        x, y, _ = ds[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        with torch.no_grad():
            p = model(x).view(1, 1, 256, 256)
            p = torch.sigmoid(p).squeeze().cpu().numpy()
            g = y.squeeze().cpu().numpy()

        stem = os.path.splitext(name)[0]
        sig = save_failure_viz(fail_dir, stem, p, g, thr_pred=thr_pred, thr_gt=args.thr_gt)

        failure_rows.append({
            "name": name,
            "mae": r["mae"],
            "rel": r["rel"],
            "ucd": r["ucd"],
            "bcd": r["bcd"],
            **sig
        })

    with open(os.path.join(args.out_dir, "failure_signatures.json"), "w") as f:
        json.dump(failure_rows, f, indent=2)

    print(f"[saved] failure visualizations + signatures to: {fail_dir}")
    print(f"[saved] {args.out_dir}/failure_signatures.json")

    # Optional: quick scatter to see “clusters” of failures (fp_rate vs fn_rate)
    if len(failure_rows) >= 2:
        xs = [row["fp_rate"] for row in failure_rows]
        ys = [row["fn_rate"] for row in failure_rows]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys)
        plt.xlabel("FP rate")
        plt.ylabel("FN rate")
        plt.title("Failure signatures (top BCD frames)")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(args.out_dir, "failure_scatter_fp_fn.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
