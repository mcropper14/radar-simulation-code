import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_preparation import get_default_params  # same as RaDelft
from compute_metrics import compute_pd_pfa
import time


from carla_radeflt_dataset_time import CarlaRADCUBE_DATASET_TIME
from network_time import NeuralNetworkRadarDetector, IN_CHANNELS, OUT_CLASSES





def compute_metrics_time_carla(
        ckpt_path,
        root="radelf_spherical_ns",
        seq_len=3,
        batch_size=1,
        num_workers=1,
        device="cuda",  
        params=None,
        quantile=0.75,
        max_scores_per_batch=200_000,
        roc_save_path="roc_carla_time_ns.png",
):



    test_dataset = CarlaRADCUBE_DATASET_TIME(
        root=root,
        params=params,
        mode="test",
        seq_len=seq_len,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"[compute_metrics_time_carla] loading checkpoint: {ckpt_path}")
    model = NeuralNetworkRadarDetector(
        arch="FPN",
        encoder_name="resnet18",
        params=params,
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
    )

  
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Helper to crop predictions to 500x240 like the loss
    def crop_pred_to_label_region(pred_tensor):
        """
        pred_tensor: (B, T, E, 512, 256) for non-BEV
        returns: (B, T, E, 500, 240)
        """
        if not params["bev"]:
            # symmetric crop to match how we padded: (top=6,bottom=6,left=8,right=8)
            return pred_tensor[:, :, :, 6:-6, 8:-8]
        else:
            # If BEV, E is spatial as well, so crop last two dims
            return pred_tensor[:, :, 6:-6, 8:-8]

   



   
    t0 = time.time()
    it = iter(test_loader)
    print("[probe] calling next(iter(test_loader)) ...")
    batch = next(it)
    print("[probe] got first batch after", time.time() - t0, "sec")


    print("==============================================")
    print(f"[Q-calibration] Estimating {int(quantile*100)}th percentile threshold...")
    all_scores = []

    first_batch_debugged = False

    with torch.no_grad():


        t0 = time.perf_counter()
        for i, (RAD_cube, gt_lidar_cube) in enumerate(test_loader):
            if i % 10 == 0:
                dt = time.perf_counter() - t0
                print(f"[progress] batch {i}/{len(test_loader)} elapsed={dt:.1f}s")


        for RAD_cube, gt_lidar_cube in test_loader:
            RAD_cube = RAD_cube.to(device)
            gt_lidar_cube = gt_lidar_cube.to(device)

            

            # Forward
            RAE_cube = model(RAD_cube)   # (B, T, E, R_pad, A_pad)
            pred = torch.sigmoid(RAE_cube)

            # Crop padding to match lidar grid (500x240)
            pred = crop_pred_to_label_region(pred)   # (B, T, E, 500, 240)

            # Debug once on first batch
            if not first_batch_debugged:
                print("==== Q-CALIBRATION DEBUG FIRST BATCH ====")
                print(" raw pred shape:", RAE_cube.shape)
                print(" cropped pred shape:", pred.shape)
                print(" raw pred stats min/max/mean:",
                      float(torch.sigmoid(RAE_cube).min()),
                      float(torch.sigmoid(RAE_cube).max()),
                      float(torch.sigmoid(RAE_cube).mean()))
                print(" cropped pred stats min/max/mean:",
                      float(pred.min()),
                      float(pred.max()),
                      float(pred.mean()))
                print(" gt_lidar_cube shape:", gt_lidar_cube.shape)
                print("=========================================")
                first_batch_debugged = True

            # Collapse over elevation: RA cell score = max over E
            # pred_scores: (B, T, 500, 240)
            pred_scores = pred.max(dim=2).values

            pred_np = pred_scores.cpu().numpy()
            flat = pred_np.reshape(-1)

            # Subsample scores per batch to avoid huge memory
            if flat.size > max_scores_per_batch:
                idx = np.random.choice(flat.size, max_scores_per_batch, replace=False)
                flat = flat[idx]

            all_scores.append(flat)

    if not all_scores:
        print("[Q-calibration] No scores collected, returning zeros.")
        return 0.0, 0.0

    all_scores = np.concatenate(all_scores, axis=0)
    thr_q = float(np.quantile(all_scores, quantile))

    print(f"[Q-calibration] collected {all_scores.size} scores")
    print(f"[Q-calibration] {int(quantile*100)}th percentile threshold = {thr_q:.4f}")
    print("==============================================")



    #Radeflt uses a 0.55 threshold
    fixed_thresholds = [0.20, 0.30, 0.50,0.55,0.70, 0.90]
    #fixed_thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    summary_thresholds = fixed_thresholds + [thr_q]
    metrics = {thr: {"pd": [], "pfa": []} for thr in summary_thresholds}

 
    roc_thresholds = np.linspace(0.0, 1.0, 101) 
    roc_pd = {thr: [] for thr in roc_thresholds}
    roc_pfa = {thr: [] for thr in roc_thresholds}

    first_batch_debugged = False

    with torch.no_grad():


        t0 = time.perf_counter()
        for i, (RAD_cube, gt_lidar_cube) in enumerate(test_loader):
            if i % 10 == 0:
                dt = time.perf_counter() - t0
                print(f"[progress] batch {i}/{len(test_loader)} elapsed={dt:.1f}s")

            RAD_cube = RAD_cube.to(device)
            gt_lidar_cube = gt_lidar_cube.to(device)

        
            RAE_cube = model(RAD_cube)      # (B, T, E, R_pad, A_pad)
            pred = torch.sigmoid(RAE_cube)

           
            pred = crop_pred_to_label_region(pred)   # (B, T, E, 500, 240)

            # Convert to numpy
            pred_np_full = pred.cpu().numpy()             # (B, T, E, 500, 240)
            gt_np_full   = gt_lidar_cube.cpu().numpy()    # (B, T, E, 500, 240)

            # Ground truth binarization in elevation dimension
            gt_bin_elev = (gt_np_full > 0).astype(np.uint8)

            # Collapse elevation: RA cell is positive if ANY elevation bin is positive
            # pred_scores: (B, T, 500, 240), gt_bin: (B, T, 500, 240)
            pred_scores = pred_np_full.max(axis=2)
            gt_bin      = gt_bin_elev.max(axis=2)

            # Debug only for first batch of second pass
            if not first_batch_debugged:
                print("==== METRICS DEBUG FIRST TEST BATCH ====")
                print(" raw pred shape:", RAE_cube.shape)
                print(" cropped pred shape:", pred_np_full.shape)
                print(" pred_scores shape (after max over E):", pred_scores.shape)
                print(" gt_bin shape (after max over E):", gt_bin.shape)
                print(" pred_scores stats min/max/mean:",
                      float(pred_scores.min()),
                      float(pred_scores.max()),
                      float(pred_scores.mean()))
                print(" gt positive fraction:", float(gt_bin.mean()))
                print("========================================")
                first_batch_debugged = True

            # --- A) your fixed + Q thresholds ---
            for thr in summary_thresholds:
                pred_bin = (pred_scores >= thr).astype(np.uint8)
                pd, pfa = compute_pd_pfa(gt_bin, pred_bin)
                metrics[thr]["pd"].append(pd)
                metrics[thr]["pfa"].append(pfa)

            # --- B) ROC sweep thresholds ---
            for thr in roc_thresholds:
                pred_bin = (pred_scores >= thr).astype(np.uint8)
                pd, pfa = compute_pd_pfa(gt_bin, pred_bin)
                roc_pd[thr].append(pd)
                roc_pfa[thr].append(pfa)

    # ---------------------------
    # 3) Aggregate & print summary thresholds
    # ---------------------------
    print("==============================================")
    print("[CARLA Time Metrics] (after crop + max_elevation)")

    for thr in fixed_thresholds:
        mean_pd = float(np.mean(metrics[thr]["pd"])) if metrics[thr]["pd"] else 0.0
        mean_pfa = float(np.mean(metrics[thr]["pfa"])) if metrics[thr]["pfa"] else 0.0
        print(f" thr={thr:0.2f} : Pd={mean_pd:.4f}, Pfa={mean_pfa:.6f}")

    # Quantile-based threshold
    mean_pd_q = float(np.mean(metrics[thr_q]["pd"])) if metrics[thr_q]["pd"] else 0.0
    mean_pfa_q = float(np.mean(metrics[thr_q]["pfa"])) if metrics[thr_q]["pfa"] else 0.0
    print(f" thr=Q{int(quantile*100)} ({thr_q:.4f}) : Pd={mean_pd_q:.4f}, Pfa={mean_pfa_q:.6f}")
    print("==============================================")

    # ---------------------------
    # 4) Build ROC curve from dense thresholds
    # ---------------------------
    roc_pd_mean = []
    roc_pfa_mean = []

    for thr in roc_thresholds:
        pd_vals = roc_pd[thr]
        pfa_vals = roc_pfa[thr]
        roc_pd_mean.append(float(np.mean(pd_vals)) if pd_vals else 0.0)
        roc_pfa_mean.append(float(np.mean(pfa_vals)) if pfa_vals else 0.0)

    roc_pd_mean = np.array(roc_pd_mean)
    roc_pfa_mean = np.array(roc_pfa_mean)

    desired_thr = 0.60
    idx = int(np.argmin(np.abs(roc_thresholds - desired_thr)))
    print(f"[At thr={roc_thresholds[idx]:.2f}] Pd={roc_pd_mean[idx]:.4f}, Pfa={roc_pfa_mean[idx]:.6f}")


    # ---------------------------
    # 5) Plot ROC curve
    # ---------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(roc_pfa_mean, roc_pd_mean, marker=None)
    plt.scatter(roc_pfa_mean, roc_pd_mean, s=5)


    summary_pfa = []
    summary_pd = []
    summary_labels = []

    for thr in fixed_thresholds:
        summary_pfa.append(float(np.mean(metrics[thr]["pfa"])))
        summary_pd.append(float(np.mean(metrics[thr]["pd"])))
        summary_labels.append(f"{thr:.2f}")

    summary_pfa.append(mean_pfa_q)
    summary_pd.append(mean_pd_q)
    summary_labels.append(f"Q{int(quantile*100)}")

    plt.scatter(summary_pfa, summary_pd, color="red", s=30, label="Fixed/Q points")

    for x, y, label in zip(summary_pfa, summary_pd, summary_labels):
        plt.text(x + 0.005, y, label, fontsize=9)

    plt.xlabel("Pfa")
    plt.ylabel("Pd")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if roc_save_path is not None:
        plt.savefig(roc_save_path, dpi=200)
        print(f"[ROC] saved ROC curve to {roc_save_path}")

    plt.show()

    return mean_pd_q, mean_pfa_q, roc_pfa_mean, roc_pd_mean


if __name__ == "__main__":
    params = get_default_params()
    params["dataset_path"] = "radelf_spherical_pc"
    params["bev"] = False

    #this is the best so far
    #ckpt_path = "lightning_logs3/radar_time/checkpoints/epoch=02-val_loss=0.0448.ckpt"
    #ckpt_path = "/home/dvij/Documents/radar_code/lightning_logs3/radar_time/checkpoints/epoch=02-val_loss=0.0881.ckpt"
    #ckpt_path = "/home/dvij/Documents/radar_code/lightning_logs3/radar_time/checkpoints/epoch=07-val_loss=0.0677.ckpt"
    
    #ckpt_path = "/home/dvij/Documents/radar_code/lightning_logs6/new_scenes/checkpoints/epoch=03-val_loss=0.3709.ckpt"
    
    
    
    
    #ckpt_path = "/home/dvij/Documents/radar_code/lightning_logs6/new_scenes_md/checkpoints/epoch=00-val_loss=0.0864.ckpt"

    #ckpt_path = "retrain/new_scenes_md/checkpoints/epoch=02-val_loss=0.0460.ckpt"

    #ckpt_path = "retraingf/new_scenes_md/checkpoints/epoch=00-val_loss=0.0116.ckpt"
    #/home/dvij/Documents/radar_code/retraingf/new_scenes_md/checkpoints/epoch=00-val_loss=0.0116.ckpt


    #ckpt_path = "retraingf/new_scenes_md/checkpoints/last.ckpt"

    #ckpt_path = "retraingf/new_scenes_md/checkpoints/epoch=00-val_loss=0.0127.ckpt"

    #ckpt_path = "retraingf/new_scenes_md/checkpoints/epoch=01-val_loss=0.0113.ckpt"

    #ckpt_path = "retraingf/new_scenes_md/checkpoints/epoch=00-val_loss=0.0418.ckpt"
    #ckpt_path = "retraingf/new_scenes_md/checkpoints/last-v2.ckpt"

    #ckpt_path = "retraingf/new_scenes_md/checkpoints/epoch=04-val_loss=0.0126.ckpt"
    #ckpt_path = "retraingf/new_scenes_md/checkpoints/last-v3.ckpt"
    ckpt_path = "retraingf/new_scenes_md/checkpoints/epoch=02-val_loss=0.0468.ckpt"

    compute_metrics_time_carla(
        ckpt_path=ckpt_path,
        root="test_lidar_images/radelf_spherical",
        seq_len=3,
        batch_size=1,
        num_workers=1,
        device="cpu", #cuda
        params=params,
        quantile=0.75,
        roc_save_path="roc_carla_time_ns.png",
    )
