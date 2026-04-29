import os
import time
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models


class SpectrumPairDataset(Dataset):
    def __init__(self, radar_dir, target_dir, transform=None, limit=None):
        self.radar_dir = radar_dir
        self.target_dir = target_dir
        self.transform = transform

        radar_files = sorted(glob.glob(os.path.join(radar_dir, "*")))
        target_files = sorted(glob.glob(os.path.join(target_dir, "*")))

     
        radar_base = {os.path.basename(f): f for f in radar_files}
        target_base = {os.path.basename(f): f for f in target_files}
        common = sorted(set(radar_base.keys()) & set(target_base.keys()))

        if limit is not None:
            common = common[:limit]

        self.pairs = [(radar_base[n], target_base[n]) for n in common]

        if len(self.pairs) == 0:
            raise RuntimeError("No matching spectrum files found between radar_dir and target_dir.")

        print(f"[SpectrumPairDataset] pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        radar_path, target_path = self.pairs[idx]


        radar = self._load_any(radar_path)
        target = self._load_any(target_path)

        if self.transform:
            radar = self.transform(radar)
            target = self.transform(target)

        return radar, target

    @staticmethod
    def _load_any(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path).astype(np.float32)
            # normalize to [0,1] if not already
            arr = arr - arr.min()
            denom = max(arr.max(), 1e-6)
            arr = arr / denom
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            return img
        else:
            return Image.open(path).convert("L")


def make_resnet101_regressor():
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # output a flat 256*256
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 256 * 256)
    return model


def train(
    radar_dir="spectrum_out/radar_depth_map_spectrum",
    target_dir="spectrum_out/lidar_depth_map_spectrum",
    out_ckpt="spectrum_out/spectrum_model_best.pt",
    batch_size=8,
    lr=1e-4,
    epochs=50,
    limit=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # (1,256,256) in [0,1]
    ])

    ds = SpectrumPairDataset(radar_dir, target_dir, transform=transform, limit=limit)

    n_train = int(0.9 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = make_resnet101_regressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)  # (B,1,256,256)
            y = y.to(device)  # (B,1,256,256)

            pred = model(x).view(-1, 1, 256, 256)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).view(-1, 1, 256, 256)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        dt = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_ckpt)

        print(f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f} time={dt:.1f}s")

    print("Saved best ckpt:", out_ckpt)

if __name__ == "__main__":
    train()
