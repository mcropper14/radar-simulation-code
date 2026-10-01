"""Train and evaluate a temporal 3D-radar occupancy model.

This is intentionally separate from ``network_time.py``.  The existing model
learns from Doppler bins; this model removes Doppler before the network and
uses only native radar occupancy plus elevation.  Both models use the same
LiDAR targets, temporal split, and output grid.

Examples:
    python3 network_3d_radar.py --train \
        --root ../test_lidar_images2/radelf_spherical \
        --checkpoint-dir ./checkpoints_3d

    python3 network_3d_radar.py --evaluate \
        --root ../test_lidar_images2/radelf_spherical \
        --checkpoint ./checkpoints_3d/last.ckpt \
        --csv ./3d_metrics.csv
"""

import argparse
import csv
import os
import time
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

try:
    from radar_pipeline.carla_radeflt_dataset_time import CarlaRADCUBE_DATASET_TIME
    from radar_pipeline.data_preparation import get_default_params
except ModuleNotFoundError:
    from carla_radeflt_dataset_time import CarlaRADCUBE_DATASET_TIME
    from data_preparation import get_default_params


OUT_CLASSES = 34
IN_CHANNELS = 64
SEQ_LEN = 3


class Radar3DDataset(CarlaRADCUBE_DATASET_TIME):
    """Convert the existing radar cube to a Doppler-free 3D-radar input."""

    def __init__(self, *args, radar_subdir="radar_spher", **kwargs):
        super().__init__(*args, radar_subdir=radar_subdir, **kwargs)

    def __getitem__(self, idx):
        radar_cube, lidar_cube = super().__getitem__(idx)
        occupancy = radar_cube[:, 0].amax(dim=1)
        power = radar_cube[:, 0].sum(dim=1)
        elevation_sum = radar_cube[:, 1].sum(dim=1)
        elevation = elevation_sum / power.clamp_min(1.0)
        radar_3d = torch.stack((occupancy, elevation), dim=1)
        return radar_3d.float(), lidar_cube.float()


class Radar3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, IN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(IN_CHANNELS),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class NeuralNetworkRadar3D(pl.LightningModule):
    """Temporal segmentation model using Doppler-free radar input."""

    def __init__(self, arch, encoder_name, params, in_channels=IN_CHANNELS,
                 out_classes=OUT_CLASSES):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
        )
        self.radar_encoder = Radar3DEncoder()
        kernel_size = (3, 5, 7)
        self.temporal = nn.Sequential(
            nn.Conv3d(OUT_CLASSES, 12, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv3d(12, 6, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv3d(6, OUT_CLASSES, kernel_size=kernel_size, padding="same"),
        )

    def forward(self, image):
        batch, steps, channels, rows, cols = image.shape
        if steps != SEQ_LEN or channels != 2:
            raise ValueError(
                f"Expected input (B,{SEQ_LEN},2,R,A), got {tuple(image.shape)}"
            )
        masks = []
        for frame in torch.unbind(image, dim=1):
            features = self.radar_encoder(frame).float()
            features = F.pad(features, (8, 8, 6, 6))
            masks.append(self.model(features))
        per_frame = torch.stack(masks, dim=1)
        temporal_input = per_frame.permute(0, 2, 1, 3, 4)
        temporal_output = self.temporal(temporal_input)
        return temporal_output.permute(0, 2, 1, 3, 4)

    def shared_step(self, batch, stage):
        radar, target = batch
        prediction = self(radar)
        prediction = prediction[:, :, :, 6:-6, 8:-8]
        loss = F.binary_cross_entropy_with_logits(
            prediction,
            target,
            pos_weight=torch.tensor([8.0], device=self.device),
        )
        self.log(
            f"{'val' if stage == 'valid' else stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=radar.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }


def make_loader(root, params, mode, batch_size, workers, radar_subdir):
    return DataLoader(
        Radar3DDataset(
            root=root,
            params=params,
            mode=mode,
            seq_len=SEQ_LEN,
            radar_subdir=radar_subdir,
        ),
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=workers,
        pin_memory=False,
    )


def pd_pfa(target, prediction):
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    positives = int(target.sum())
    negatives = int((~target).sum())
    return (
        float((target & prediction).sum() / positives) if positives else 0.0,
        float((~target & prediction).sum() / negatives) if negatives else 0.0,
    )


def load_model(checkpoint, params, device):
    model = NeuralNetworkRadar3D(
        arch="FPN",
        encoder_name="resnet18",
        params=params,
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    state_dict = saved.get("state_dict", saved)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def evaluate(
    checkpoint,
    root,
    params,
    device,
    batch_size,
    workers,
    threshold,
    radar_subdir,
):
    loader = make_loader(
        root, params, "test", batch_size, workers, radar_subdir
    )
    model = load_model(checkpoint, params, device)

    pd_values, pfa_values = [], []
    with torch.no_grad():
        for radar, target in loader:
            scores = torch.sigmoid(model(radar.to(device)))
            scores = scores[:, :, :, 6:-6, 8:-8].amax(dim=2).cpu().numpy()
            target = target.numpy().max(axis=2)
            for score, truth in zip(scores.reshape(-1, *scores.shape[-2:]),
                                     target.reshape(-1, *target.shape[-2:])):
                pd, pfa = pd_pfa(truth, score >= threshold)
                pd_values.append(pd)
                pfa_values.append(pfa)

    summary = {
        "frames": len(pd_values),
        "threshold": threshold,
        "pd_mean": float(np.mean(pd_values)) if pd_values else 0.0,
        "pfa_mean": float(np.mean(pfa_values)) if pfa_values else 0.0,
        "pd_std": float(np.std(pd_values)) if pd_values else 0.0,
        "pfa_std": float(np.std(pfa_values)) if pfa_values else 0.0,
    }
    print("3D radar model")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def train(args, params):
    train_loader = make_loader(
        args.root, params, "train", args.batch_size, args.workers, args.radar_subdir
    )
    val_loader = make_loader(
        args.root, params, "val", args.batch_size, args.workers, args.radar_subdir
    )
    model = NeuralNetworkRadar3D(
        arch="FPN", encoder_name="resnet18", params=params,
        in_channels=IN_CHANNELS, out_classes=OUT_CLASSES,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="3d-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[callback],
        log_every_n_steps=5,
    )
    started = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"Training time: {(time.time() - started) / 60:.2f} minutes")
    print(f"Best checkpoint: {callback.best_model_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_3d")
    parser.add_argument("--csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--radar-subdir",
        default="radar_spher",
        help="Radar directory under ROOT; use radar_4d_spher for modified radar.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.evaluate and not args.checkpoint:
        parser.error("--checkpoint is required with --evaluate")
    params = get_default_params()
    params["dataset_path"] = args.root
    params["bev"] = False
    if args.train:
        train(args, params)
    else:
        summary = evaluate(
            args.checkpoint, args.root, params, args.device,
            args.batch_size, args.workers, args.threshold, args.radar_subdir,
        )
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=summary.keys())
                writer.writeheader()
                writer.writerow(summary)


if __name__ == "__main__":
    main()
