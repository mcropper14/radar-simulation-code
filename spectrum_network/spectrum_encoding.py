import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch

from data_preparation import get_default_params
from carla_radeflt_dataset_time import CarlaRADCUBE_DATASET_TIME



#implement the matlab code from algorithm.m 
def corr2(a: np.ndarray, b: np.ndarray) -> float:
    """MATLAB corr2 equivalent."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.min(x)
    m = np.max(x)
    if m < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return x / m


def log_compress_and_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.log10(np.maximum(x, eps)).astype(np.float32)
    y = y + abs(np.min(y))
    return normalize01(y)



def estimate_spectrum(image: np.ndarray, M: int, theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """
    Python port of the MATLAB estimate().

    MATLAB:
      basis_phi  = exp(-j*pi*m*sin(phi));
      basis_theta = exp(-j*pi*m*sin(theta));
      C_phi  = basis_phi' * basis_phi;
      C_theta = basis_theta' * basis_theta;

      P(jj,ii) = sum(abs(image .* transpose(elevation*azimuth)), 'all');

    The key simplification:
      sum(abs(image[p,t] * C_phi[jj,p] * C_theta[ii,t]))
    can be vectorized as:
      P = abs(C_phi) @ abs(image) @ abs(C_theta).T
    """
    H = len(phi_deg)
    W = len(theta_deg)
    img = cv2.resize(image.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

    m = np.arange(1, M + 1,dtype=np.float32)[:, None]  #(M,1)

    theta = np.deg2rad(theta_deg.astype(np.float32))[None, :]  #(1,W)
    phi = np.deg2rad(phi_deg.astype(np.float32))[None, :]  #(1,H)

    #(M,W) and (M,H)
    basis_theta = np.exp(-1j* np.pi * m * np.sin(theta))  #(M,W)
    basis_phi = np.exp(-1j* np.pi * m * np.sin(phi)) #(M,H)

    #covaraince matrices
    C_theta = basis_theta.conj().T @ basis_theta  #(W,W)
    C_phi = basis_phi.conj().T @ basis_phi #(H,H)


    P = (np.abs(C_phi).astype(np.float32) @ np.abs(img).astype(np.float32) @ np.abs(C_theta).astype(np.float32).T)
    return P.astype(np.float32)




def radar_cube_to_range_az_map(RAD_cube_one_frame: torch.Tensor) -> np.ndarray:

    rad = RAD_cube_one_frame.detach().cpu().numpy() 
    occ = rad[0]  

    ra = occ.sum(axis=0)  # (R,A)
    return ra.astype(np.float32)


def lidar_cube_to_range_az_map(GT_cube_one_frame: torch.Tensor) -> np.ndarray:

    gt = GT_cube_one_frame.detach().cpu().numpy()  # (E,R,A)
    ra = gt.sum(axis=0)  # (R,A)
    return ra.astype(np.float32)



def main():
    params = get_default_params()
    params["dataset_path"] = "radelf_spherical_ns2"  
    ds = CarlaRADCUBE_DATASET_TIME(root=params["dataset_path"], params=params, mode="train", seq_len=1)

    out_dir = "spectrum_sanity_out"
    os.makedirs(out_dir, exist_ok=True)

    """

    % Algorithm parameters
        M_camera = 200;
        M_radar = 20;
        theta = -70:1:70;
        phi = -70:1:70;
    """



    #from matlab
    M_camera = 200
    M_radar = 20
    theta = np.arange(-70, 71, 1, dtype=np.float32)  # az
    phi = np.arange(-70, 71, 1, dtype=np.float32)    # "el" axis in paper

   
    n = min(10, len(ds))
    corr_orig = []
    corr_spec = []

    for i in range(n):
        RAD_seq, GT_seq = ds[i]             
        RAD = RAD_seq[0]                    
        GT = GT_seq[0]                      

        radar_img = radar_cube_to_range_az_map(RAD)   
        lidar_img = lidar_cube_to_range_az_map(GT)    

   
        radar_img_256 = cv2.resize(radar_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        lidar_img_256 = cv2.resize(lidar_img, (256, 256), interpolation=cv2.INTER_NEAREST)

        
        P_radar = estimate_spectrum(radar_img_256, M_radar, theta, phi)
        P_lidar = estimate_spectrum(lidar_img_256, M_camera, theta, phi)

      
        P_radar_n = log_compress_and_normalize(P_radar)
        P_lidar_n = log_compress_and_normalize(P_lidar)

      
        c0 = corr2(radar_img_256, lidar_img_256)
        c1 = corr2(P_radar_n, P_lidar_n)
        corr_orig.append(c0)
        corr_spec.append(c1)

      
        fig = plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        plt.title("LiDAR Occupancy (Range×Az)")
        plt.imshow(normalize01(lidar_img_256), aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Az bins")
        plt.ylabel("Range bins")

        ax2 = plt.subplot(1, 2, 2)
        plt.title("Radar Occupancy (Range×Az)")
        plt.imshow(normalize01(radar_img_256), aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Az bins")
        plt.ylabel("Range bins")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"occ_pair_{i:02d}.png"), dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Spectrum(LiDAR)")
        plt.imshow(P_lidar_n, aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 2, 2)
        plt.title("Spectrum(Radar)")
        plt.imshow(P_radar_n, aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"spectrum_pair_{i:02d}.png"), dpi=200)
        plt.close(fig)

        print(f"[{i:02d}] corr original={c0:.3f} | corr spectrum={c1:.3f}")

    print("\n=== Mean correlations ===")
    print(f"Mean corr original: {np.mean(corr_orig):.3f}")
    print(f"Mean corr spectrum: {np.mean(corr_spec):.3f}")
    print(f"\nSaved figures to: {out_dir}/")


if __name__ == "__main__":
    main()
