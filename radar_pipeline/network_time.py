#https://github.com/RaDelft/RaDelft-Dataset/blob/main/machine_learning_python/networks/network_time.py


import time
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

sys.path.append(sys.path[0] + "/..")

from data_preparation import get_default_params  
from carla_radeflt_dataset_time import CarlaRADCUBE_DATASET_TIME


#match all the radelft configurations
OUT_CLASSES = 34  # 44 elevation angles      
IN_CHANNELS = 64   # output of the ReduceDNet
    

#I don't think this changes any speed
torch.set_float32_matmul_precision("medium")


DEBUG_ON = True
DEBUG_EVERY_N = 500   # print every N training steps, inc this to 500



class DopplerEncoder(nn.Module):
    def __init__(self):
        super(DopplerEncoder, self).__init__()

        in_channels = 2           # [elevation, power] channels in Doppler cube
        out_channel_1 = 32
        out_channel_2 = IN_CHANNELS
        kernel_size1 = (5, 3, 3)
        stride1 = (4, 1, 1)   #reduce Doppler by 4
        padding1 = (2, 1, 1)
        kernel_size2 = (4, 3, 3)
        stride2 = (4, 1, 1)   #reduce Doppler by another 4
        padding2 = (1, 1, 1)
        pool_kernel = (8, 1, 1)  #collapse Doppler to 1
        pool_stride = (8, 1, 1)

        self.conv1 = nn.Conv3d(in_channels, out_channel_1,kernel_size=kernel_size1,stride=stride1,padding=padding1)
        self.norm1 = nn.BatchNorm3d(out_channel_1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(out_channel_1, out_channel_2,kernel_size=kernel_size2,stride=stride2,padding=padding2)
        self.norm2 = nn.BatchNorm3d(out_channel_2)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool3d(kernel_size=pool_kernel,stride=pool_stride)

    def forward(self, x):
        # x: (B, 2, D, R, A)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.pool(x)  #(B, IN_CHANNELS, 1, R, A)
        x = x.squeeze(2)  #remove Doppler dim to (B, IN_CHANNELS, R, A)
        return x


class NeuralNetworkRadarDetector(pl.LightningModule):
    def __init__(self, arch, encoder_name, params, in_channels, out_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.params = params
        self.DopplerReducer = DopplerEncoder()

        #2D segmentation backbone over (R, A)
        self.model = smp.create_model(
            arch,encoder_name=encoder_name,in_channels=in_channels,classes=out_classes,
            **kwargs,
        )

        #Temporal 3D conv over time / class / R / A
        kernel_size = (3, 5, 7)
        
        self.conv1 = nn.Conv3d(3, 6, kernel_size=kernel_size, padding="same")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(6, 12, kernel_size=kernel_size, padding="same")
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(12, 6, kernel_size=kernel_size, padding="same")
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(6, 3, kernel_size=kernel_size, padding="same")

    def forward(self, image):
        
        #image: (B, T, 2, D, R, A)
        
        
        B, T, C, D, R, A = image.shape
        #radelft t = 3 
        assert T == 3, f"Current implementation assumes T=3, got T={T}"

        # Split over time: list of (B, 1, 2, D, R, A)
        frames = torch.chunk(image, T, dim=1)

        masks_per_t = []
        for t_idx, frame in enumerate(frames):
            # (B,1,2,D,R,A) to (B,2,D,R,A)
            frame = frame.squeeze(1)

            # Doppler encoder: (B,2,D,R,A) to (B,IN_CHANNELS,R,A)
            feat = self.DopplerReducer(frame).float()

            # Pad to (R,A) = (512,256) for FPN divisibility by 32
            # Current lidar/radar grid is (500,240)
            feat = F.pad(feat, pad=(8, 8, 6, 6))  # (left,right,top,bottom)
            #(B, IN_CHANNELS, 512, 256)

            # 2D segmentation per frame
            mask_t = self.model(feat)  # (B, OUT_CLASSES, 512, 256)
            masks_per_t.append(mask_t)

        # Stack over time: (B, OUT_CLASSES, 512,256, T)
        mask = torch.stack(masks_per_t, dim=4)

        # Permute to (B, T, OUT_CLASSES, 512, 256)
        mask = mask.permute(0, 4, 1, 2, 3)

        # Temporal smoothing with 3D conv
        x = mask  # (B, 3, OUT_CLASSES, 512, 256)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)  # still (B,3,OUT_CLASSES,512,256)

        # Return in the same layout as loss expects: (B, T, OUT_CLASSES, 512,256)
        return x

    def shared_step(self, batch, stage: str):
        """
        Common logic for train/val/test.
        batch: (RAD_cube, gt_lidar_cube)
          RAD_cube:      (B, T, 2, D, R, A)
          gt_lidar_cube: (B, T, E, 500, 240)
        """
        RAD_cube, gt_lidar_cube = batch

        # Forward pass
        RAE_cube = self.forward(RAD_cube)  # (B, T, OUT_CLASSES, 512,256)

        # Crop to match 500x240 label grid match radelft 
        RAE_crop = RAE_cube[:, :, :, 6:-6, 8:-8]  # (B, T, OUT_CLASSES, 500, 240)
        gt = gt_lidar_cube.float()

        #try increasing loss weigth value
        #increase to 8.0
        pos_weight = torch.tensor([8.0], device=self.device)
        loss = F.binary_cross_entropy_with_logits(
            RAE_crop,
            gt,
            pos_weight=pos_weight,
        )

        if DEBUG_ON:
            step_idx = int(self.global_step) if hasattr(self, "global_step") else 0
            do_print = False
            if stage == "train" and step_idx % DEBUG_EVERY_N == 0:
                do_print = True
            if stage == "valid" and step_idx % DEBUG_EVERY_N == 0:
                do_print = True

            if do_print:
                #test and look at the gt fraction
                with torch.no_grad():
                    print(f"\n[DEBUG {stage}] ")
                    print(f"  RAD_cube shape: {tuple(RAD_cube.shape)}")
                    print(f"  gt_lidar shape: {tuple(gt_lidar_cube.shape)}")
                    print(f"  RAE_cube shape: {tuple(RAE_cube.shape)}")
                    print(f"  RAE_crop shape: {tuple(RAE_crop.shape)}")

                    pred_sig = torch.sigmoid(RAE_crop)
                    print(
                        f"  pred_sig stats: "
                        f"min={pred_sig.min().item():.4f}, "
                        f"max={pred_sig.max().item():.4f}, "
                        f"mean={pred_sig.mean().item():.4f}"
                    )
                    gt_pos_frac = (gt > 0.5).float().mean().item()
                    print(f"  GT positive fraction: {gt_pos_frac:.6f}")
                    print("  -----------------------------------------\n")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "valid")
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch[0].shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main(params):
    root = params["dataset_path"]

    # Datasets
    train_dataset = CarlaRADCUBE_DATASET_TIME(
        root=root,
        params=params,
        mode="train",
        seq_len=3,
    )
    val_dataset = CarlaRADCUBE_DATASET_TIME(
        root=root,
        params=params,
        mode="val",
        seq_len=3,
    )

    print(f"train len = {len(train_dataset)}")
    print(f"val len = {len(val_dataset)}")

    #debug
    RAD_sample, GT_sample = train_dataset[0]
    print(f"sample RAD shape: {RAD_sample.shape}  GT shape: {GT_sample.shape}")

    # DataLoaders
    num_workers = 2  #cannot increase over 2 on 1070., they used 8 in example
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

 
    model = NeuralNetworkRadarDetector(
        arch="FPN",
        encoder_name="resnet18",
        params=params,
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="retraingf/new_scenes_md/checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        verbose=True,
    )
    #this gives an error, don't use
    progress_bar = RichProgressBar(
        leave=True,
        theme=RichProgressBarTheme(metrics_format=".4e"),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=20,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    start_time = time.time()
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Training time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")

    print("Best model path:", checkpoint_callback.best_model_path)
    print("Best val_loss:", checkpoint_callback.best_model_score.item())


if __name__ == "__main__":
    params = get_default_params()
    
    params["dataset_path"] = "test_lidar_images2/radelf_spherical"
    params["bev"] = False

    main(params)
