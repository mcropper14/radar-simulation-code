import os
import glob
import numpy as np
import cv2


ROOT = "radelf_spherical_ns2"  
RADAR_DIR = os.path.join(ROOT, "radar_spher")
LIDAR_DIR = os.path.join(ROOT, "lidar_spher")

OUT_DIR = "spectrum_out"
OUT_RAD_SPEC = os.path.join(OUT_DIR, "radar_depth_map_spectrum")
OUT_LID_SPEC = os.path.join(OUT_DIR, "lidar_depth_map_spectrum")
OUT_DEBUG = os.path.join(OUT_DIR, "debug_figs")
os.makedirs(OUT_RAD_SPEC, exist_ok=True)
os.makedirs(OUT_LID_SPEC, exist_ok=True)
os.makedirs(OUT_DEBUG, exist_ok=True)

# MATLAB parameters
M_RADAR = 20
M_LIDAR = 200  # they used larger M for camera; we use it for LiDAR-proxy spectrum
theta_deg = np.arange(-70, 71, 1)  # azimuth bins (deg)
phi_deg = np.arange(-70, 71, 1)    # elevation bins (deg)

N_FILES = 10          # sanity run; set None to do all
SAVE_DEBUG_EVERY = 1  # save debug images for first N frames

# ----------------------------
# Helpers
# ----------------------------
def corr2(a: np.ndarray, b: np.ndarray) -> float:
    """MATLAB-like corr2."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    if denom < 1e-12:
        return 0.0
    return float((a.ravel() @ b.ravel()) / denom)

def crop_spherical_to_roi(spher: np.ndarray, roi_spherical):
    """
    roi_spherical = [r_min, az_min_deg, el_min_deg, r_max, az_max_deg, el_max_deg]
    input spher: [r, az(rad), el(rad), doppler]
    """
    if spher.size == 0:
        return spher
    r_min, az_min_d, el_min_d, r_max, az_max_d, el_max_d = roi_spherical
    az = spher[:, 1]
    el = spher[:, 2]
    r = spher[:, 0]

    az_min = np.deg2rad(az_min_d)
    az_max = np.deg2rad(az_max_d)
    el_min = np.deg2rad(el_min_d)
    el_max = np.deg2rad(el_max_d)

    m = (
        (r >= r_min) & (r <= r_max) &
        (az >= az_min) & (az <= az_max) &
        (el >= el_min) & (el <= el_max)
    )
    return spher[m]

def spherical_to_depth_map(spher: np.ndarray, theta_deg, phi_deg, r_max_fill=120.0):
    """
    Build a depth map on (phi x theta) grid:
      rows: elevation (phi), cols: azimuth (theta)
    We store nearest range per bin (min r), and fill empty with 0.
    """
    H = len(phi_deg)
    W = len(theta_deg)
    depth = np.zeros((H, W), dtype=np.float32)

    if spher.size == 0:
        return depth

    r = spher[:, 0].astype(np.float32)
    az = np.rad2deg(spher[:, 1]).astype(np.float32)
    el = np.rad2deg(spher[:, 2]).astype(np.float32)

    # bin indices
    az_idx = np.round(az - theta_deg[0]).astype(np.int32)
    el_idx = np.round(el - phi_deg[0]).astype(np.int32)

    valid = (az_idx >= 0) & (az_idx < W) & (el_idx >= 0) & (el_idx < H)
    az_idx = az_idx[valid]
    el_idx = el_idx[valid]
    r = r[valid]

    # nearest range per (el, az)
    # if multiple hits land in same cell, keep min range
    for ei, ai, ri in zip(el_idx, az_idx, r):
        if depth[ei, ai] == 0.0:
            depth[ei, ai] = ri
        else:
            depth[ei, ai] = min(depth[ei, ai], ri)

    # optional: clamp very large values
    depth = np.clip(depth, 0.0, r_max_fill)
    return depth

def estimate_spectrum(image_2d: np.ndarray, M: int, theta_deg, phi_deg) -> np.ndarray:
    """
    Python port of the MATLAB `estimate()`.

    image_2d is resized to (len(phi), len(theta)).
    Produces P of shape (len(phi), len(theta)).
    """
    theta = np.deg2rad(theta_deg.astype(np.float64))
    phi = np.deg2rad(phi_deg.astype(np.float64))
    H = len(phi)
    W = len(theta)

    img = cv2.resize(image_2d.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)

    m = np.arange(1, M + 1, dtype=np.float64)[:, None]  # (M,1)

    # basis: (M, len(phi/theta))
    basis_phi = np.exp(-1j * np.pi * (m * np.sin(phi)[None, :]))
    basis_theta = np.exp(-1j * np.pi * (m * np.sin(theta)[None, :]))

    # covariance-like: (len(phi),len(phi)) and (len(theta),len(theta))
    C_phi = (basis_phi.conj().T @ basis_phi)      # (H,H)
    C_theta = (basis_theta.conj().T @ basis_theta)  # (W,W)

    P = np.zeros((H, W), dtype=np.float64)

    # direct port of their nested loops (still fine at 141x141)
    # P[jj, ii] = sum(abs(img .* transpose(elevation*azimuth)), 'all')
    for ii in range(W):
        elevation = C_theta[ii, :].reshape(W, 1)  # (W,1)
        for jj in range(H):
            azimuth = C_phi[jj, :].reshape(1, H)  # (1,H)
            # elevation*azimuth => (W,H) ; transpose => (H,W)
            kernel = (elevation @ azimuth).T      # (H,W)
            P[jj, ii] = np.sum(np.abs(img * kernel))

    return P.astype(np.float32)

def compress_normalize(P: np.ndarray) -> np.ndarray:
    """
    Matches their preprocessing:
      resize -> log10 -> shift -> normalize to [0,1]
    """
    P = cv2.resize(P.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)

    # avoid log(0)
    P = np.maximum(P, 1e-12)
    P = np.log10(P)

    # shift to be non-negative then normalize
    P = P - P.min()
    mx = float(P.max())
    if mx > 1e-12:
        P = P / mx
    return P.astype(np.float32)

def save_gray_png(path: str, img01: np.ndarray):
    """Save [0,1] float as 8-bit PNG."""
    im8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, im8)

# ----------------------------
# Main preprocessing
# ----------------------------
def main():
    # Use your ROI if you want; otherwise keep full grid range.
    # If you want to match your preprocessing params, set these:
    roi_spherical = [0, -70, -15, 50, 70, 15]  # from your defaults (range, az, el)

    radar_files = sorted(glob.glob(os.path.join(RADAR_DIR, "*.npy")))
    lidar_files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.npy")))

    radar_base = set(os.path.basename(f) for f in radar_files)
    lidar_base = set(os.path.basename(f) for f in lidar_files)
    common = sorted(list(radar_base & lidar_base))

    if N_FILES is not None:
        common = common[:N_FILES]

    corr_original = []
    corr_spectrum = []

    for i, fname in enumerate(common):
        radar_spher = np.load(os.path.join(RADAR_DIR, fname))  # (Nr,4)
        lidar_spher = np.load(os.path.join(LIDAR_DIR, fname))  # (Nl,4)

        radar_spher = crop_spherical_to_roi(radar_spher, roi_spherical)
        lidar_spher = crop_spherical_to_roi(lidar_spher, roi_spherical)

        # depth maps (phi x theta)
        depth_radar = spherical_to_depth_map(radar_spher, theta_deg, phi_deg)
        depth_lidar = spherical_to_depth_map(lidar_spher, theta_deg, phi_deg)

        # "original correlation" (proxy for their depth_map vs semantic)
        corr_o = corr2(depth_radar, depth_lidar)
        corr_original.append(corr_o)

        # spectrum estimate
        P_radar = estimate_spectrum(depth_radar, M_RADAR, theta_deg, phi_deg)
        P_lidar = estimate_spectrum(depth_lidar, M_LIDAR, theta_deg, phi_deg)

        # compress & normalize
        P_radar_n = compress_normalize(P_radar)
        P_lidar_n = compress_normalize(P_lidar)

        corr_s = corr2(P_radar_n, P_lidar_n)
        corr_spectrum.append(corr_s)

        # save images
        save_gray_png(os.path.join(OUT_RAD_SPEC, f"{i:06d}.png"), P_radar_n)
        save_gray_png(os.path.join(OUT_LID_SPEC, f"{i:06d}.png"), P_lidar_n)

        if i < SAVE_DEBUG_EVERY:
            # quick debug collage
            dbg = np.zeros((256, 256 * 2), dtype=np.uint8)
            dbg[:, :256] = (P_lidar_n * 255).astype(np.uint8)
            dbg[:, 256:] = (P_radar_n * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT_DEBUG, f"dbg_{i:03d}.png"), dbg)

        print(f"[{i:02d}] corr original={corr_o:+.3f} | corr spectrum={corr_s:+.3f}")

    print("\n=== Mean correlations ===")
    print(f"Mean corr original: {np.mean(corr_original):.3f}")
    print(f"Mean corr spectrum: {np.mean(corr_spectrum):.3f}")
    print(f"\nWrote:\n  {OUT_RAD_SPEC}\n  {OUT_LID_SPEC}\n  {OUT_DEBUG}")

if __name__ == "__main__":
    main()
