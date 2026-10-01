
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


from data_preparation import (
    get_default_params,
    lidarpc_to_lidarcube,
    spherical_to_cartesian,
)




def crop_spherical_to_roi(spher: np.ndarray, params: dict) -> np.ndarray:
    """
    Crop spherical points [r, az(rad), el(rad), ...] to the radar ROI.

    params['roi_spherical'] = [r_min, az_min_deg, el_min_deg, r_max, az_max_deg, el_max_deg]
    """
    if spher.size == 0:
        return spher

    roi = params["roi_spherical"]
    r_min, az_min_d, el_min_d, r_max, az_max_d, el_max_d = roi

    az_min = np.deg2rad(az_min_d)
    az_max = np.deg2rad(az_max_d)
    el_min = np.deg2rad(el_min_d)
    el_max = np.deg2rad(el_max_d)

    r = spher[:, 0]
    az = spher[:, 1]
    el = spher[:, 2]

    mask = (
        (r >= r_min) & (r <= r_max) &
        (az >= az_min) & (az <= az_max) &
        (el >= el_min) & (el <= el_max)
    )
    return spher[mask]


def points_to_radar_cube_from_spherical(radar_spher: np.ndarray, params: dict) -> np.ndarray:
    """
    Build a radar cube directly from spherical radar points.

    radar_spher: (N, 4) -> [range(m), az(rad), el(rad), doppler(m/s)]
    params must contain:
        'range_axis'   (len R = 500)
        'azimuth_axis' (len A = 240, radians)
        'vel_fft_size' (D = 128)
        'vel_bin_size' (m/s per Doppler bin)

    Returns:
        radar_cube : (2, D, R, A) float32
            channel 0: occupancy / power (binary 0/1)
            channel 1: average elevation per voxel (rad)
    """
    range_axis = params["range_axis"]
    azimuth_axis = params["azimuth_axis"]
    vel_fft_size = params["vel_fft_size"]
    vel_bin_size = params["vel_bin_size"]

    D = vel_fft_size
    R = len(range_axis)
    A = len(azimuth_axis)

    # Empty cube if no points
    if radar_spher.size == 0:
        return np.zeros((2, D, R, A), dtype=np.float32)

    r = radar_spher[:, 0]
    az = radar_spher[:, 1]
    el = radar_spher[:, 2]
    v = radar_spher[:, 3]

    # Bin range and azimuth
    r_idx = np.searchsorted(range_axis, r, side="left")
    a_idx = np.searchsorted(azimuth_axis, az, side="left")

    # Clip into grid
    R_max = R - 1
    A_max = A - 1
    r_idx = np.clip(r_idx, 0, R_max)
    a_idx = np.clip(a_idx, 0, A_max)

    # Doppler -> D index, centered at vel_fft_size/2
    d_idx = np.round(v / vel_bin_size + vel_fft_size / 2.0).astype(int)
    d_idx = np.clip(d_idx, 0, vel_fft_size - 1)

    radar_cube = np.zeros((2, D, R, A), dtype=np.float32)

    # Channel 0: occupancy/power
    radar_cube[0, d_idx, r_idx, a_idx] = 1.0

    # Channel 1: average elevation per voxel
    el_sum = np.zeros((D, R, A), dtype=np.float32)
    el_count = np.zeros((D, R, A), dtype=np.float32)

    for di, ri, ai, eli in zip(d_idx, r_idx, a_idx, el):
        el_sum[di, ri, ai] += eli
        el_count[di, ri, ai] += 1.0

    nonzero = el_count > 0
    radar_cube[1][nonzero] = el_sum[nonzero] / el_count[nonzero]

    return radar_cube.astype(np.float32)


class CarlaRADCUBE_DATASET_TIME(Dataset):
    """
    Time-window radar/lidar cubes from CARLA spherical npys.

    Directory layout (root):

        radelf_spherical_pc/
            lidar_spher/
                0.000000000.npy   # (N_l, 4) [r, az, el, 0]
                0.133333333.npy
                ...
            radar_spher/
                0.000000000.npy   # (N_r, 4) [r, az, el, v_r]
                0.133333333.npy
                ...

    Each __getitem__ returns:

        RAD_cube : (T, 2, D, 500, 240)
        GT_cube  : (T, 34, 500, 240)

    where T = seq_len.
    """

    def __init__(
        self, root, params=None, mode="train", seq_len=3,
        radar_subdir="radar_spher",
    ):
        super().__init__()
        self.root = root
        self.params = params if params is not None else get_default_params()
        self.seq_len = seq_len

        self.lidar_dir = os.path.join(root, "lidar_spher")
        self.radar_dir = os.path.join(root, radar_subdir)

        lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.npy")))
        radar_files = sorted(glob.glob(os.path.join(self.radar_dir, "*.npy")))

        lidar_base = [os.path.basename(f) for f in lidar_files]
        radar_base = [os.path.basename(f) for f in radar_files]

        # Frames that have BOTH lidar and radar
        common = sorted(set(lidar_base) & set(radar_base))
        self.frames = common

        # Build sliding windows of length seq_len
        self.samples = []
        for i in range(len(self.frames) - seq_len + 1):
            window = self.frames[i : i + seq_len]
            self.samples.append(window)

        n = len(self.samples)

        if mode == "train":
            self.samples = self.samples[: int(0.8 * n)]
        elif mode == "val":
            self.samples = self.samples[int(0.8 * n) : int(0.9 * n)]
        elif mode == "test":
            self.samples = self.samples[int(0.9 * n) :]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No time windows found in {self.root} "
                f"for mode={mode} (frames={len(self.frames)}, seq_len={self.seq_len})"
            )

        print(
            f"[CarlaRADCUBE_DATASET_TIME] root={self.root}, mode={mode}, "
            f"frames={len(self.frames)}, windows={len(self.samples)}, seq_len={self.seq_len}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_names = self.samples[idx]  # list of length seq_len

        radar_seq = []
        lidar_seq = []

        for fname in frame_names:

            lidar_spher = np.load(os.path.join(self.lidar_dir, fname))  # (N_l, 4)

            # crop to ROI
            lidar_spher = crop_spherical_to_roi(lidar_spher, self.params)

            if lidar_spher.size == 0:
                # empty cube (E, R, A)
                E = len(self.params["elevation_axis"])
                R = len(self.params["range_axis"])
                A = len(self.params["azimuth_axis"])
                lidar_cube = np.zeros((E, R, A), dtype=np.float32)
            else:
                r_l = lidar_spher[:, 0]
                az_l = lidar_spher[:, 1]
                el_l = lidar_spher[:, 2]

                # back to cartesian for lidarpc_to_lidarcube
                lidar_xyz = spherical_to_cartesian(r_l, az_l, el_l)  # (N_l, 3)
                lidar_cube = lidarpc_to_lidarcube(lidar_xyz, self.params)  # (E, R, A)

   
            radar_spher = np.load(os.path.join(self.radar_dir, fname))  # (N_r, 4)

            radar_spher = crop_spherical_to_roi(radar_spher, self.params)
            radar_cube = points_to_radar_cube_from_spherical(radar_spher, self.params)  # (2, D, R, A)

            radar_seq.append(radar_cube)
            lidar_seq.append(lidar_cube)

        radar_seq = np.stack(radar_seq, axis=0)  # (T, 2, D, R, A)
        lidar_seq = np.stack(lidar_seq, axis=0)  # (T, E, R, A)

        RAD_cube = torch.from_numpy(radar_seq.astype(np.float32))
        GT_cube = torch.from_numpy(lidar_seq.astype(np.float32))

        return RAD_cube, GT_cube




def get_carla_params():
    """
    Simple helper if you want a named function.
    Currently just returns the RaDelft defaults;
    you can override dataset_path if you want.
    """
    params = get_default_params()
    params["dataset_path"] = "radelf_spherical_ns2"
    return params
