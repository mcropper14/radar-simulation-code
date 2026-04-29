"""
Modified for the 3d point cloud generation
"""


"""

python3 spectrum_resnset.py \
  --x_dir spectrum_out/radar_depth_map_spectrum \
  --y_dir spectrum_out/mutual_spectrum_gt \
  --epochs 50 --batch_size 8 --lr 1e-4




"""


#!/usr/bin/env python3
import os
import glob
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Dataset
# -------------------------
class SpectrumPairDataset(Dataset):
    """
    X: radar_depth_map_spectrum (1x256x256)
    Y: mutual_spectrum_gt       (1x256x256)
    Matches by filename stem.
    """
    def __init__(self, x_dir, y_dir, transform=None, limit=None):
        self.transform = transform

        x_files = sorted(glob.glob(os.path.join(x_dir, "*.png")))
        y_files = sorted(glob.glob(os.path.join(y_dir, "*.png")))

        x_map = {os.path.basename(f): f for f in x_files}
        y_map = {os.path.basename(f): f for f in y_files}
        common = sorted(set(x_map.keys()) & set(y_map.keys()))

        if limit is not None:
            common = common[:limit]

        if len(common) == 0:
            raise RuntimeError(
                f"No matching png files.\nX: {x_dir}\nY: {y_dir}\n"
                "Make sure filenames match exactly."
            )

        self.names = common
        self.pairs = [(x_map[n], y_map[n]) for n in common]
        print(f"[SpectrumPairDataset] pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.pairs[idx]
        name = self.names[idx]

        x = Image.open(x_path).convert("L")
        y = Image.open(y_path).convert("L")

        if self.transform:
            x = self.transform(x)  # (1,256,256)
            y = self.transform(y)
        else:
            x = transforms.ToTensor()(x)
            y = transforms.ToTensor()(y)

        return x, y, name


# -------------------------
# Model
# -------------------------
def make_resnet101_regressor(out_hw=256):
    """
    ResNet101 backbone, 1-channel input, regression head to 256*256.
    """
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_hw * out_hw)
    return model


# -------------------------
# Helpers
# -------------------------
def save_img01(path, img01):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img01 = np.clip(img01, 0.0, 1.0)
    im8 = (img01 * 255).astype(np.uint8)
    Image.fromarray(im8).save(path)


def plot_curves(out_path, train_losses, val_losses):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Train
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--x_dir", default="spectrum_out/radar_depth_map_spectrum")
    ap.add_argument("--y_dir", default="spectrum_out/mutual_spectrum_gt")

    ap.add_argument("--out_dir", default="spectrum_out")
    ap.add_argument("--ckpt_best", default="spectrum_model_best.pt")
    ap.add_argument("--ckpt_last", default="spectrum_model_last.pt")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_debug", type=int, default=10, help="save first N val predictions each epoch (0 disables)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, args.ckpt_best)
    last_path = os.path.join(args.out_dir, args.ckpt_last)

    # Transform: keep in [0,1], ensure 256x256
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # (1,256,256) float in [0,1]
    ])

    ds = SpectrumPairDataset(args.x_dir, args.y_dir, transform=tfm, limit=args.limit)

    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = make_resnet101_regressor(out_hw=256).to(device)

    # Since target is [0,1], you can either:
    # (A) regression directly + clamp later
    # (B) apply sigmoid to outputs (recommended so output stays bounded)
    use_sigmoid = True

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    train_losses, val_losses = [], []

    print(f"[split] train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        tr_sum = 0.0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)  # (B,1,256,256)
            yb = yb.to(device, non_blocking=True)  # (B,1,256,256)

            pred = model(xb).view(-1, 1, 256, 256)
            if use_sigmoid:
                pred = torch.sigmoid(pred)

            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_sum += loss.item() * xb.size(0)

        tr_loss = tr_sum / len(train_loader.dataset)
        train_losses.append(tr_loss)

        # ---- val ----
        model.eval()
        va_sum = 0.0

        # optional debug output folder
        dbg_dir = os.path.join(args.out_dir, "train_debug", f"epoch_{epoch:03d}")
        saved = 0

        with torch.no_grad():
            for xb, yb, names in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb).view(-1, 1, 256, 256)
                if use_sigmoid:
                    pred = torch.sigmoid(pred)

                loss = criterion(pred, yb)
                va_sum += loss.item() * xb.size(0)

                # save a few debug triplets
                if args.save_debug > 0 and saved < args.save_debug:
                    xb_np = xb.cpu().numpy()[:, 0]
                    yb_np = yb.cpu().numpy()[:, 0]
                    pr_np = pred.cpu().numpy()[:, 0]

                    for i in range(pr_np.shape[0]):
                        if saved >= args.save_debug:
                            break
                        name = os.path.splitext(names[i])[0]
                        save_img01(os.path.join(dbg_dir, f"{name}_in.png"), xb_np[i])
                        save_img01(os.path.join(dbg_dir, f"{name}_gt.png"), yb_np[i])
                        save_img01(os.path.join(dbg_dir, f"{name}_pred.png"), pr_np[i])
                        saved += 1

        va_loss = va_sum / len(val_loader.dataset)
        val_losses.append(va_loss)

        # ---- checkpoints ----
        torch.save(model.state_dict(), last_path)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] train={tr_loss:.6f} val={va_loss:.6f} best={best_val:.6f} time={dt:.1f}s")

        # update curves plot each epoch
        plot_curves(os.path.join(args.out_dir, "train_curves.png"), train_losses, val_losses)

    print("\nDone.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("Curves:", os.path.join(args.out_dir, "train_curves.png"))


if __name__ == "__main__":
    main()


