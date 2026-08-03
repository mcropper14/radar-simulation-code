"""
Closed-loop radar-to-control pipeline in CARLA.

Deploys the trained 4D radar detector inside CARLA in real time, feeds its
detections into a simple ACC-style controller, and logs closed-loop metrics
(false-brake events, TTC, following distance, inference latency) for the
IROS 2026 sim-to-real workshop paper.

Usage:
    python3 closed_loop_runner.py --ckpt /path/to/checkpoint.ckpt --frames 500
    python3 closed_loop_runner.py --ckpt /path/to/checkpoint.ckpt --frames 500 --device cpu
    python3 closed_loop_runner.py --ckpt /path/to/checkpoint.ckpt --frames 500 --out-dir ./cl_run1

Architecture:
    CARLA radar sensor
        → raw_radar_to_cube()          # ego-motion Doppler, spherical binning
        → deque(maxlen=3)              # rolling temporal buffer
        → NeuralNetworkRadarDetector   # (1,3,2,D,R,A) → (1,3,34,512,256)
        → sigmoid + crop + threshold   # 0.55
        → decode_detections()          # voxels → Cartesian XYZ + velocity
        → acc_controller()             # proportional brake / cruise
        → ego.apply_control()          # closed loop
        → metrics logger               # CSV per tick
"""

import os
import csv
import math
import time
import random
import argparse
import collections
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

CARLA_AVAILABLE = False
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    pass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths — adjust sys.path so imports work when run from any directory
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(__file__))

from data_preparation import get_default_params
from carla_radeflt_dataset_time import (
    crop_spherical_to_roi,
    points_to_radar_cube_from_spherical,
    get_carla_params,
)
from network_time import NeuralNetworkRadarDetector, IN_CHANNELS, OUT_CLASSES

# ---------------------------------------------------------------------------
# CARLA connection
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 2002
TIMEOUT_S = 10.0

FPS = 20.0
DT = 1.0 / FPS

TOWN_NAME = "Town04"
EGO_VEHICLE_BP = "vehicle.tesla.model3"
EGO_ROLE_NAME = "ego"
LEAD_VEHICLE_BP = "vehicle.audi.a2"

# Sensor mounting positions (match new_delft2.py exactly)
LIDAR_TRANSFORM_CONFIG = {"x": 0.0, "y": 0.0, "z": 2.2, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
RADAR_TRANSFORM_CONFIG = {"x": 2.0, "y": 0.0, "z": 1.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}

RADAR_HORIZONTAL_FOV = 70
RADAR_VERTICAL_FOV = 20
RADAR_RANGE = 100.0
RADAR_PPS = 500_000

WARMUP_FRAMES = 40
DEBUG_PRINT_EVERY = 30

# ACC controller parameters
ACC_SAFE_DISTANCE_M = 15.0     # minimum following distance (m)
ACC_CRUISE_THROTTLE = 0.45     # default throttle when no obstacle
ACC_MAX_BRAKE = 0.8            # maximum brake force
ACC_KP_DIST = 0.05             # proportional gain on distance error
ACC_KP_VEL = 0.2               # proportional gain on closing velocity
ACC_FORWARD_ARC_Y = 3.5        # half-width of the forward detection lane (m)
ACC_MIN_RANGE_M = 2.0          # ignore detections closer than this (ego body)
ACC_FALSE_BRAKE_THRESHOLD = 0.1   # brake > this counts as a "brake event"
ACC_FALSE_BRAKE_GT_RANGE = 25.0   # no real obstacle if gt_range > this (m)

# Closed-start frames before inference is valid (need seq_len=3 buffer)
SEQ_LEN = 3
COLD_START_FRAMES = SEQ_LEN - 1   # 2 frames of default cruise before first inference

# Detection threshold (match RaDelft convention)
DETECTION_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Sensor buffer
# ---------------------------------------------------------------------------
@dataclass
class SensorBuffer:
    radar_pts_sensor: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Helper: rotation matrix ego → world (yaw only)
# ---------------------------------------------------------------------------
def _R_ego_to_world(yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 1: raw radar → (2, D, R, A) cube
# Reproduces save_spherical_frame's radar path exactly — ego-motion Doppler
# projection (NOT CARLA's native .velocity) to match the training dataset.
# ---------------------------------------------------------------------------
def raw_radar_to_cube(
    radar_raw: np.ndarray,          # (N, 4) [depth, azimuth, altitude, velocity_native]
    radar_ego_xyz: np.ndarray,      # (N, 3) Cartesian in ego frame
    ego_pos: np.ndarray,            # (3,) world position
    ego_yaw: float,                 # degrees
    last_ego_pos: Optional[np.ndarray],
    last_time: Optional[float],
    t_now: float,
    params: dict,
) -> np.ndarray:
    """
    Build a (2, D, R, A) radar cube in memory from a single tick of raw radar
    detections. Uses ego-motion Doppler projection to match training data.

    Returns zeros cube if no radar points.
    """
    if radar_ego_xyz is None or len(radar_ego_xyz) == 0:
        D = params["vel_fft_size"]
        R = len(params["range_axis"])
        A = len(params["azimuth_axis"])
        return np.zeros((2, D, R, A), dtype=np.float32)

    # Ego velocity in world frame (finite difference — same as training)
    if last_ego_pos is not None and last_time is not None:
        dt = max(t_now - last_time, 1e-6)
        ego_vel_world = (ego_pos - last_ego_pos) / dt
    else:
        ego_vel_world = np.zeros(3, np.float32)

    # Doppler: project ego velocity onto each radar ray direction in world frame
    R_e2w = _R_ego_to_world(ego_yaw)
    radar_pts_world = (R_e2w @ radar_ego_xyz.T).T

    norms = np.linalg.norm(radar_pts_world, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    u_r = radar_pts_world / norms

    # Doppler = -ego_vel · unit_radial  (positive = approaching)
    v_rel_world = -ego_vel_world[None, :]
    doppler_vals = np.sum(v_rel_world * u_r, axis=1).astype(np.float32)

    # Spherical coordinates from Cartesian ego-frame points
    x = radar_ego_xyz[:, 0]
    y = radar_ego_xyz[:, 1]
    z = radar_ego_xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    r_safe = np.maximum(r, 1e-6)
    el = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))

    radar_spher = np.stack([r, az, el, doppler_vals], axis=1).astype(np.float32)

    # Crop to ROI and bin into cube
    radar_spher = crop_spherical_to_roi(radar_spher, params)
    radar_cube = points_to_radar_cube_from_spherical(radar_spher, params)
    return radar_cube


# ---------------------------------------------------------------------------
# Step 3: Decode model output → Cartesian detections + velocity
# ---------------------------------------------------------------------------
def decode_detections(
    pred_cropped: np.ndarray,   # (34, 500, 240) sigmoid output, already thresholded
    input_cube: np.ndarray,     # (2, D, R, A) raw radar cube for velocity recovery
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert thresholded model output into Cartesian XYZ detections and velocities.

    Returns:
        xyz   : (M, 3) Cartesian positions in ego frame (m)
        vels  : (M,)  radial velocities (m/s, positive = approaching)
    """
    range_axis = params["range_axis"]
    azimuth_axis = params["azimuth_axis"]
    elevation_axis = params["elevation_axis"]
    vel_fft_size = params["vel_fft_size"]
    vel_bin_size = params["vel_bin_size"]

    # Binary occupancy: max over elevation → BEV-like (R, A)
    bev = np.max(pred_cropped, axis=0)          # (500, 240)
    occ = bev > 0                               # binary mask

    if not np.any(occ):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

    r_idx, a_idx = np.where(occ)

    # Guard against out-of-range indices (cube may be padded)
    R_cube = input_cube.shape[2]
    A_cube = input_cube.shape[3]
    r_idx_clipped = np.clip(r_idx, 0, R_cube - 1)
    a_idx_clipped = np.clip(a_idx, 0, A_cube - 1)

    # Range / azimuth from axis arrays
    range_values = range_axis[np.clip(r_idx, 0, len(range_axis) - 1)]
    azimuth_values = azimuth_axis[np.clip(a_idx, 0, len(azimuth_axis) - 1)]

    # Elevation: take elevation bin with max occupancy for this (r, a) cell
    el_idx = np.argmax(pred_cropped[:, r_idx, a_idx], axis=0)
    elevation_values = elevation_axis[np.clip(el_idx, 0, len(elevation_axis) - 1)]

    # Velocity: argmax over Doppler dim of channel-0 input cube at (r, a).
    # Index [0] first to get a 3D array (D, R, A), then advanced-index (r, a)
    # so that (D, M) shape is preserved correctly before the argmax.
    d_idx = np.argmax(input_cube[0][:, r_idx_clipped, a_idx_clipped], axis=0)
    vel_values = (d_idx.astype(np.float32) - vel_fft_size / 2.0) * vel_bin_size

    # Spherical → Cartesian
    x = range_values * np.cos(elevation_values) * np.cos(azimuth_values)
    y = range_values * np.cos(elevation_values) * np.sin(azimuth_values)
    z = range_values * np.sin(elevation_values)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    return xyz, vel_values.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Simple ACC controller
# ---------------------------------------------------------------------------
def acc_controller(
    detections_xyz: np.ndarray,   # (M, 3) Cartesian ego frame
    velocities: np.ndarray,       # (M,) radial velocity (m/s)
    ego_velocity_ms: float,       # ego forward speed (m/s)
) -> "carla.VehicleControl":
    """
    Proportional ACC controller: brake if lead vehicle detected within safe
    distance or has positive closing velocity; otherwise cruise.

    Controller design is deliberately simple — the contribution is the
    closed-loop *measurement*, not controller sophistication.
    """
    control = carla.VehicleControl()

    # Filter to forward arc only
    if len(detections_xyz) > 0:
        forward_mask = (
            (detections_xyz[:, 0] > ACC_MIN_RANGE_M) &
            (np.abs(detections_xyz[:, 1]) < ACC_FORWARD_ARC_Y)
        )
        fwd_xyz = detections_xyz[forward_mask]
        fwd_vel = velocities[forward_mask] if len(velocities) > 0 else np.array([])
    else:
        fwd_xyz = np.empty((0, 3))
        fwd_vel = np.array([])

    if len(fwd_xyz) == 0:
        # No forward obstacle — cruise
        control.throttle = ACC_CRUISE_THROTTLE
        control.brake = 0.0
        control.steer = 0.0
        return control

    # Closest forward detection
    ranges = np.linalg.norm(fwd_xyz[:, :2], axis=1)
    idx = np.argmin(ranges)
    min_range = float(ranges[idx])
    closing_vel = float(fwd_vel[idx]) if len(fwd_vel) > 0 else 0.0

    dist_error = min_range - ACC_SAFE_DISTANCE_M   # positive = too far
    brake_cmd = 0.0

    if dist_error < 0:
        # Too close — hard proportional brake
        brake_cmd = min(ACC_MAX_BRAKE, -ACC_KP_DIST * dist_error)
    elif closing_vel > 0:
        # Approaching — soft brake
        brake_cmd = min(ACC_MAX_BRAKE * 0.5, ACC_KP_VEL * closing_vel)

    if brake_cmd > 0.05:
        control.throttle = 0.0
        control.brake = float(np.clip(brake_cmd, 0.0, ACC_MAX_BRAKE))
    else:
        control.throttle = ACC_CRUISE_THROTTLE
        control.brake = 0.0

    control.steer = 0.0
    return control


# ---------------------------------------------------------------------------
# Sensor callbacks
# ---------------------------------------------------------------------------
def radar_callback(meas: "carla.RadarMeasurement", buf: SensorBuffer):
    rows = []
    for d in meas:
        rows.append((float(d.depth), float(d.azimuth), float(d.altitude), float(d.velocity)))
    buf.radar_pts_sensor = np.array(rows, dtype=np.float32) if rows else np.empty((0, 4), dtype=np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def radar_spherical_to_xyz(r_az_alt: np.ndarray) -> np.ndarray:
    """Convert (N, 3) [depth, azimuth, altitude] → Cartesian."""
    if r_az_alt is None or len(r_az_alt) == 0:
        return np.empty((0, 3), dtype=np.float32)
    r = r_az_alt[:, 0]
    az = r_az_alt[:, 1]
    alt = r_az_alt[:, 2]
    x = r * np.cos(alt) * np.cos(az)
    y = r * np.cos(alt) * np.sin(az)
    z = r * np.sin(alt)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points_sensor_to_world(sensor_tf: "carla.Transform", pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    M = np.array(sensor_tf.get_matrix(), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), np.ones((len(pts), 1), np.float32)], axis=1)
    return ((M @ pts_h.T).T)[:, :3]


def world_points_to_ego(ego_tf: "carla.Transform", pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    inv = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), np.ones((len(pts), 1), np.float32)], axis=1)
    return ((inv @ pts_h.T).T)[:, :3]


def get_actor_range_velocity(ego_tf: "carla.Transform", lead: "carla.Actor") -> Tuple[float, float]:
    """
    Oracle: ground-truth range and closing velocity from CARLA actor transforms.
    Used ONLY for evaluation metrics — never as controller input.
    """
    lead_loc = lead.get_location()
    lead_world = np.array([lead_loc.x, lead_loc.y, lead_loc.z], dtype=np.float32)
    lead_ego = world_points_to_ego(ego_tf, lead_world[None, :])[0]
    gt_range = float(np.linalg.norm(lead_ego[:2]))

    ego_v = ego_tf.get_forward_vector()
    lead_v_carla = lead.get_velocity()
    ego_speed = math.sqrt(ego_v.x**2 + ego_v.y**2)
    lead_speed = math.sqrt(lead_v_carla.x**2 + lead_v_carla.y**2)
    # Approximate closing velocity (positive = approaching)
    gt_closing_vel = ego_speed - lead_speed

    return gt_range, gt_closing_vel


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, params: dict, device: str) -> torch.nn.Module:
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
    print(f"[load_model] Loaded checkpoint: {ckpt_path} → {device}")
    return model


# ---------------------------------------------------------------------------
# Main closed-loop run
# ---------------------------------------------------------------------------
def run_closed_loop(
    ckpt_path: str,
    num_frames: int,
    out_dir: str,
    device: str,
    spawn_index: int,
    lead_decel_frame: int,
):
    if not CARLA_AVAILABLE:
        raise RuntimeError("CARLA Python API not found.")

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "closed_loop_metrics.csv")

    params = get_carla_params()

    print("[run_closed_loop] Loading model...")
    model = load_model(ckpt_path, params, device)

    # Offline sanity check: single forward pass on dummy input
    D = params["vel_fft_size"]        # 128
    R = len(params["range_axis"])     # 500 (before padding to 512)
    A = len(params["azimuth_axis"])   # 240 (before padding to 256)
    _dummy = torch.zeros(1, SEQ_LEN, 2, D, 512, 256, device=device)
    with torch.no_grad():
        _out = model(_dummy)
    print(f"[run_closed_loop] Model sanity check passed: output shape {tuple(_out.shape)}")
    del _dummy, _out

    # Rolling temporal buffer
    frame_buffer: collections.deque = collections.deque(maxlen=SEQ_LEN)

    # CARLA setup
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world = client.load_world(TOWN_NAME)
    bp_lib = world.get_blueprint_library()
    old_settings = world.get_settings()

    actors = []
    sensors = []

    try:
        # Synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        # ---- Spawn ego vehicle ----
        ego_bp = bp_lib.find(EGO_VEHICLE_BP)
        ego_bp.set_attribute("role_name", EGO_ROLE_NAME)
        spawn_points = world.get_map().get_spawn_points()
        if spawn_index >= len(spawn_points):
            spawn_index = 0
        ego = world.try_spawn_actor(ego_bp, spawn_points[spawn_index])
        if ego is None:
            for sp in spawn_points:
                ego = world.try_spawn_actor(ego_bp, sp)
                if ego is not None:
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")
        actors.append(ego)

        # ---- Spawn lead vehicle ~20 m ahead ----
        lead_bp = bp_lib.find(LEAD_VEHICLE_BP)
        ego_fwd = spawn_points[spawn_index].get_forward_vector()
        lead_loc = spawn_points[spawn_index].location
        lead_loc.x += ego_fwd.x * 20.0
        lead_loc.y += ego_fwd.y * 20.0
        lead_spawn = carla.Transform(lead_loc, spawn_points[spawn_index].rotation)
        lead = world.try_spawn_actor(lead_bp, lead_spawn)
        if lead is None:
            print("[warn] Could not spawn lead vehicle at offset; trying adjacent spawn point.")
            idx2 = (spawn_index + 1) % len(spawn_points)
            lead = world.try_spawn_actor(lead_bp, spawn_points[idx2])
        if lead is not None:
            actors.append(lead)
            lead.set_autopilot(True, traffic_manager.get_port())
            # Lead vehicle cruises slightly slower than ego
            traffic_manager.vehicle_percentage_speed_difference(lead, 10)
        else:
            print("[warn] Lead vehicle not spawned — oracle metrics will be unavailable.")

        # ---- Radar sensor ----
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(RADAR_HORIZONTAL_FOV))
        radar_bp.set_attribute("vertical_fov", str(RADAR_VERTICAL_FOV))
        radar_bp.set_attribute("range", str(RADAR_RANGE))
        radar_bp.set_attribute("points_per_second", str(RADAR_PPS))
        radar_tf = carla.Transform(
            carla.Location(**{k: RADAR_TRANSFORM_CONFIG[k] for k in ("x", "y", "z")}),
            carla.Rotation(**{k: RADAR_TRANSFORM_CONFIG[k] for k in ("pitch", "yaw", "roll")}),
        )
        buf = SensorBuffer()
        radar_sensor = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)
        sensors.append(radar_sensor)
        actors.append(radar_sensor)
        radar_sensor.listen(lambda m: radar_callback(m, buf))

        # Autopilot for ego (we will override with apply_control each tick)
        ego.set_autopilot(False)

        # Warm up
        print(f"[run_closed_loop] Warming up ({WARMUP_FRAMES} frames)...")
        for _ in range(WARMUP_FRAMES):
            world.tick()

        last_ego_pos: Optional[np.ndarray] = None
        last_time: Optional[float] = None
        cold_start_remaining = COLD_START_FRAMES

        print(f"[run_closed_loop] Starting closed-loop run ({num_frames} frames)...")

        with open(metrics_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "frame", "t_ms",
                "inference_ms",
                "n_detections",
                "detected_range_m", "detected_vel_ms",
                "gt_range_m", "gt_closing_vel_ms",
                "throttle", "brake", "steer",
                "ego_speed_ms",
                "false_brake",
                "cold_start",
            ])

            for frame in range(num_frames):
                world.tick()
                t_now = (frame + WARMUP_FRAMES) * DT

                ego_tf = ego.get_transform()
                ego_loc = ego_tf.location
                ego_pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z], dtype=np.float32)
                ego_yaw = ego_tf.rotation.yaw
                ego_v_carla = ego.get_velocity()
                ego_speed_ms = math.sqrt(ego_v_carla.x**2 + ego_v_carla.y**2 + ego_v_carla.z**2)

                # Scripted deceleration event for lead vehicle
                if lead is not None and frame == lead_decel_frame:
                    traffic_manager.vehicle_percentage_speed_difference(lead, 80)
                    print(f"[frame {frame}] Lead vehicle decelerating (scripted event).")
                if lead is not None and frame == lead_decel_frame + 100:
                    traffic_manager.vehicle_percentage_speed_difference(lead, 10)

                # ---- Build radar cube ----
                radar_raw = buf.radar_pts_sensor
                if radar_raw is not None and len(radar_raw) > 0:
                    radar_xyz_sensor = radar_spherical_to_xyz(radar_raw[:, :3])
                    radar_world = transform_points_sensor_to_world(radar_sensor.get_transform(), radar_xyz_sensor)
                    radar_ego_xyz = world_points_to_ego(ego_tf, radar_world)
                else:
                    radar_ego_xyz = np.empty((0, 3), dtype=np.float32)

                cube = raw_radar_to_cube(
                    radar_raw=radar_raw,
                    radar_ego_xyz=radar_ego_xyz,
                    ego_pos=ego_pos,
                    ego_yaw=ego_yaw,
                    last_ego_pos=last_ego_pos,
                    last_time=last_time,
                    t_now=t_now,
                    params=params,
                )
                frame_buffer.append(cube)

                # ---- Oracle ground-truth (evaluation only) ----
                gt_range_m = float("nan")
                gt_closing_vel_ms = float("nan")
                if lead is not None:
                    gt_range_m, gt_closing_vel_ms = get_actor_range_velocity(ego_tf, lead)

                # ---- Cold-start handling ----
                if cold_start_remaining > 0:
                    cold_start_remaining -= 1
                    control = carla.VehicleControl()
                    control.throttle = ACC_CRUISE_THROTTLE
                    control.brake = 0.0
                    control.steer = 0.0
                    ego.apply_control(control)

                    writer.writerow([
                        frame, f"{t_now * 1000:.1f}",
                        "nan", 0,
                        "nan", "nan",
                        f"{gt_range_m:.3f}" if not math.isnan(gt_range_m) else "nan",
                        f"{gt_closing_vel_ms:.3f}" if not math.isnan(gt_closing_vel_ms) else "nan",
                        f"{control.throttle:.3f}", f"{control.brake:.3f}", f"{control.steer:.3f}",
                        f"{ego_speed_ms:.3f}",
                        0, 1,
                    ])
                    last_ego_pos = ego_pos.copy()
                    last_time = t_now
                    continue

                # ---- Online inference ----
                t_inf_start = time.perf_counter()

                # Stack buffer → (1, T, 2, D, R_pad, A_pad)
                cubes_np = list(frame_buffer)
                stacked = np.stack(cubes_np, axis=0)  # (T, 2, D, R, A)
                # Pad R→512, A→256 (match model's padded input expectation)
                R_raw = stacked.shape[3]
                A_raw = stacked.shape[4]
                R_pad = 512
                A_pad = 256
                pad_r = max(0, R_pad - R_raw)
                pad_a = max(0, A_pad - A_raw)
                if pad_r > 0 or pad_a > 0:
                    stacked = np.pad(stacked, ((0,0),(0,0),(0,0),(0,pad_r),(0,pad_a)))

                rad_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)  # (1,T,2,512,256)

                with torch.no_grad():
                    pred = model(rad_tensor)                   # (1, T, 34, 512, 256)

                # Use the last time-step prediction
                pred_last = pred[0, -1]                        # (34, 512, 256)
                pred_last = torch.sigmoid(pred_last)
                pred_last_np = pred_last.cpu().numpy()

                # Crop to (34, 500, 240)
                pred_cropped = pred_last_np[:, 6:-6, 8:-8]    # (34, 500, 240)
                pred_cropped[pred_cropped < DETECTION_THRESHOLD] = 0.0

                t_inf_end = time.perf_counter()
                inference_ms = (t_inf_end - t_inf_start) * 1000.0

                # ---- Decode detections ----
                xyz_detections, vel_detections = decode_detections(
                    pred_cropped, cube, params
                )

                n_detections = len(xyz_detections)

                # Closest forward detection for controller + logging
                detected_range_m = float("nan")
                detected_vel_ms = float("nan")
                if n_detections > 0:
                    fwd_mask = (
                        (xyz_detections[:, 0] > ACC_MIN_RANGE_M) &
                        (np.abs(xyz_detections[:, 1]) < ACC_FORWARD_ARC_Y)
                    )
                    if np.any(fwd_mask):
                        fwd_xyz = xyz_detections[fwd_mask]
                        fwd_vel = vel_detections[fwd_mask]
                        r_fwd = np.linalg.norm(fwd_xyz[:, :2], axis=1)
                        i_min = int(np.argmin(r_fwd))
                        detected_range_m = float(r_fwd[i_min])
                        detected_vel_ms = float(fwd_vel[i_min])

                # ---- ACC controller ----
                control = acc_controller(xyz_detections, vel_detections, ego_speed_ms)
                ego.apply_control(control)

                # ---- False-brake detection ----
                false_brake = int(
                    control.brake > ACC_FALSE_BRAKE_THRESHOLD and
                    (math.isnan(gt_range_m) or gt_range_m > ACC_FALSE_BRAKE_GT_RANGE)
                )

                # ---- Log metrics ----
                writer.writerow([
                    frame, f"{t_now * 1000:.1f}",
                    f"{inference_ms:.2f}",
                    n_detections,
                    f"{detected_range_m:.3f}" if not math.isnan(detected_range_m) else "nan",
                    f"{detected_vel_ms:.3f}" if not math.isnan(detected_vel_ms) else "nan",
                    f"{gt_range_m:.3f}" if not math.isnan(gt_range_m) else "nan",
                    f"{gt_closing_vel_ms:.3f}" if not math.isnan(gt_closing_vel_ms) else "nan",
                    f"{control.throttle:.3f}", f"{control.brake:.3f}", f"{control.steer:.3f}",
                    f"{ego_speed_ms:.3f}",
                    false_brake, 0,
                ])

                if frame % DEBUG_PRINT_EVERY == 0:
                    print(
                        f"[{frame:05d}/{num_frames}] "
                        f"inf={inference_ms:.1f}ms | "
                        f"dets={n_detections} | "
                        f"det_r={detected_range_m:.1f}m | "
                        f"gt_r={gt_range_m:.1f}m | "
                        f"brake={control.brake:.2f} | "
                        f"false_brake={false_brake} | "
                        f"speed={ego_speed_ms:.1f}m/s"
                    )

                last_ego_pos = ego_pos.copy()
                last_time = t_now

        print(f"\n[run_closed_loop] Done. Metrics saved to {metrics_path}")

    finally:
        print("\n[run_closed_loop] Cleaning up...")
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass
        destroyed = set()
        for a in reversed(actors):
            try:
                if a is None or a.id in destroyed:
                    continue
                a.destroy()
                destroyed.add(a.id)
            except Exception:
                pass
        try:
            world.apply_settings(old_settings)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop 4D radar detector + ACC controller in CARLA")
    p.add_argument("--ckpt", required=True, help="Path to trained NeuralNetworkRadarDetector checkpoint")
    p.add_argument("--frames", type=int, default=500, help="Number of simulation frames to run")
    p.add_argument("--out-dir", default="./closed_loop_output", help="Directory for metrics CSV")
    p.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
    p.add_argument("--spawn-index", type=int, default=0, help="Ego vehicle spawn point index")
    p.add_argument("--lead-decel-frame", type=int, default=200,
                   help="Frame at which the lead vehicle decelerates (scripted event)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[warn] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    run_closed_loop(
        ckpt_path=args.ckpt,
        num_frames=args.frames,
        out_dir=args.out_dir,
        device=args.device,
        spawn_index=args.spawn_index,
        lead_decel_frame=args.lead_decel_frame,
    )
