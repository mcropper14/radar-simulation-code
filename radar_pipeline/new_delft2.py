"""
Outputs:
  <out-dir>/radelf_spherical/lidar_spher/<timestamp>.npy
  <out-dir>/radelf_spherical/radar_spher/<timestamp>.npy (native CARLA radar)
  <out-dir>/radelf_spherical/radar_4d_spher/<timestamp>.npy (optional modified radar)
  <out-dir>/radelf_spherical/camera_rgb/<timestamp>.png
  <out-dir>/overlay_frames/overlay_<frame>.png (if --visualize)
  <out-dir>/coverage_diagnostics.txt (real-time coverage stats)
  <out-dir>/coverage_analysis/ (if --analyze)

Usage:
  # Data collection
  python3 new_delft2.py --record --frames 2000 --out-dir ./training_data
  python3 new_delft2.py --record --visualize --viz-every 30 --frames 2000
  
  # Analyze collected data
  python3 new_delft2.py --analyze ./training_data --max-files 50
"""

import os
import csv
import json
import math
import random
import argparse
import time
from dataclasses import dataclass
from typing import Optional
from glob import glob

import numpy as np
import cv2

try:
    from radar_pipeline.radar_physics import RadarPhysicsConfig, apply_radar_physics
except ModuleNotFoundError:
    from radar_physics import RadarPhysicsConfig, apply_radar_physics


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARLA_AVAILABLE = False
try:
    import carla
    CARLA_AVAILABLE = hasattr(carla, "Client")
except ImportError:
    pass

#Do not change the ip/port configs

HOST = "127.0.0.1"
PORT = 2002
TIMEOUT_S = 10.0

FPS = 20.0
DT = 1.0 / FPS

# Environment settings
TOWN_NAME = "Town04"  
NUM_VEHICLES = 200 #600
NUM_PEDESTRIANS = 90
WEATHER_ID = 1  # 0=ClearNoon, 1=CloudyNoon, etc.

# Ego vehicle
EGO_VEHICLE_BP = "vehicle.tesla.model3"
EGO_ROLE_NAME = "ego"
USE_AUTOPILOT = True


MANUAL_DRIVE_THROTTLE = 0.6
MANUAL_DRIVE_BRAKE = 0.0


LIDAR_TRANSFORM_CONFIG = {"x": 0.0, "y": 0.0, "z": 2.2, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
RADAR_TRANSFORM_CONFIG = {"x": 2.0, "y": 0.0, "z": 1.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
CAM_TRANSFORM_CONFIG   = {"x": 1.5, "y": 0.0, "z": 1.6, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}

# Camera settings
CAM_W = 1280
CAM_H = 720
CAM_FOV = 90


LIDAR_RANGE = 120.0
LIDAR_CHANNELS = 128
LIDAR_PPS = 1200000
LIDAR_ROT_HZ = 20.0  
LIDAR_UPPER_FOV = 10.0
LIDAR_LOWER_FOV = -30.0
LIDAR_HORIZONTAL_FOV = 360.0  


RADAR_HORIZONTAL_FOV = 70
RADAR_VERTICAL_FOV = 20
RADAR_RANGE = 100.0
RADAR_PPS = 500000

#don't start at frame 1 
WARMUP_FRAMES = 40
DEBUG_PRINT_EVERY = 30
DIAGNOSTIC_COVERAGE_CHECK = True

#debug
OVERLAY_POINT_RADIUS = 2
OVERLAY_LIDAR_COLOR = (0, 255, 0)  # Green
OVERLAY_RADAR_COLOR = (255, 0, 0)  # Blue (BGR)
OVERLAY_DOWNSAMPLE_LIDAR = 4000
OVERLAY_DOWNSAMPLE_RADAR = 1000



@dataclass
class SensorBuffer:
    lidar_pts_sensor: Optional[np.ndarray] = None
    radar_pts_sensor: Optional[np.ndarray] = None
    rgb_image: Optional[np.ndarray] = None



def cartesian_to_spherical_xyz(pts):
    """
    pts: (N,3) in some Cartesian frame [x,y,z]
    returns: (N,3) [range, azimuth, elevation]
      range in meters, azimuth/elevation in radians
    """
    if pts.size == 0:
        return np.zeros((0, 3), np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    r_safe = np.maximum(r, 1e-6)
    el = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))
    return np.stack([r, az, el], axis=1)


def ts_name(t_now):
    """ROS-style timestamp: seconds.nanoseconds"""
    sec = int(t_now)
    nsec = int(round((t_now - sec) * 1e9))
    return f"{sec}.{nsec:09d}"


def R_ego_to_world(yaw_deg):
    """Rotation matrix from ego frame to world frame using yaw about z."""
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)


def check_azimuth_coverage(pts_ego_xyz, frame_num):
    """
    Diagnostic: Check if LiDAR points cover all azimuth angles (360°).
    """
    if pts_ego_xyz is None or len(pts_ego_xyz) == 0:
        return None
    
    sph = cartesian_to_spherical_xyz(pts_ego_xyz)
    azimuth_rad = sph[:, 1]
    azimuth_deg = np.degrees(azimuth_rad)
    azimuth_deg = (azimuth_deg + 180) % 360 - 180
    
    quadrants = {
        'Front': np.sum((azimuth_deg >= -45) & (azimuth_deg <= 45)),
        'Right': np.sum((azimuth_deg > 45) & (azimuth_deg <= 135)),
        'Back': np.sum((azimuth_deg > 135) | (azimuth_deg < -135)),
        'Left': np.sum((azimuth_deg >= -135) & (azimuth_deg < -45))
    }
    
    return {
        'frame': frame_num,
        'total_points': len(pts_ego_xyz),
        'azimuth_min': azimuth_deg.min(),
        'azimuth_max': azimuth_deg.max(),
        'azimuth_range': azimuth_deg.max() - azimuth_deg.min(),
        'quadrants': quadrants
    }



def transform_points_sensor_to_world(sensor_tf: "carla.Transform", pts_sensor_xyz: np.ndarray):
    """Transform points from sensor frame to world frame."""
    if pts_sensor_xyz is None or len(pts_sensor_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)
    M = np.array(sensor_tf.get_matrix(), dtype=np.float32)
    pts_h = np.concatenate([pts_sensor_xyz.astype(np.float32),
                            np.ones((len(pts_sensor_xyz), 1), dtype=np.float32)], axis=1)
    pw = (M @ pts_h.T).T
    return pw[:, :3].astype(np.float32)


def world_points_to_ego(ego_tf: "carla.Transform", pts_world_xyz: np.ndarray):
    """Transform points from world frame to ego frame."""
    if pts_world_xyz is None or len(pts_world_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)
    inv = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32)
    pts_h = np.concatenate([pts_world_xyz.astype(np.float32),
                            np.ones((len(pts_world_xyz), 1), dtype=np.float32)], axis=1)
    pe = (inv @ pts_h.T).T
    return pe[:, :3].astype(np.float32)


def radar_spherical_to_xyz(r_az_alt):
    """Convert radar spherical coordinates to Cartesian."""
    if r_az_alt is None or len(r_az_alt) == 0:
        return np.empty((0, 3), dtype=np.float32)
    r = r_az_alt[:, 0]
    az = r_az_alt[:, 1]
    alt = r_az_alt[:, 2]
    x = r * np.cos(alt) * np.cos(az)
    y = r * np.cos(alt) * np.sin(az)
    z = r * np.sin(alt)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def build_camera_K(image_w, image_h, fov_deg):
    """Build camera intrinsic matrix."""
    f = (image_w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = image_w / 2.0
    cy = image_h / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float32)


def world_to_camera_frame(cam_tf: "carla.Transform", pts_world_xyz: np.ndarray) -> np.ndarray:
    """Transform world points to camera frame."""
    if pts_world_xyz is None or len(pts_world_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)
    inv = np.array(cam_tf.get_inverse_matrix(), dtype=np.float32)
    pts_h = np.concatenate([pts_world_xyz.astype(np.float32),
                            np.ones((len(pts_world_xyz), 1), dtype=np.float32)], axis=1)
    pc = (inv @ pts_h.T).T
    return pc[:, :3].astype(np.float32)


def project_camera_xyz_to_pixels(pts_cam_xyz: np.ndarray, K: np.ndarray, W: int, H: int):
    """Project camera frame points to image pixels."""
    if pts_cam_xyz is None or len(pts_cam_xyz) == 0:
        return np.empty((0, 2), dtype=np.float32)

    X = pts_cam_xyz[:, 0]
    Y = pts_cam_xyz[:, 1]
    Z = pts_cam_xyz[:, 2]

    front = X > 0.1
    if not np.any(front):
        return np.empty((0, 2), dtype=np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * (Y[front] / X[front]) + cx
    v = fy * (-Z[front] / X[front]) + cy

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uv = np.stack([u[inside], v[inside]], axis=1).astype(np.float32)
    return uv


def downsample_points(pts: np.ndarray, target_n: int) -> np.ndarray:
    """Downsample points for visualization."""
    if pts is None or len(pts) == 0:
        return np.empty((0, pts.shape[1] if pts.ndim == 2 else 3), dtype=np.float32)
    if len(pts) <= target_n:
        return pts.astype(np.float32)
    indices = np.random.choice(len(pts), target_n, replace=False)
    return pts[indices].astype(np.float32)


def draw_points_on_image(img: np.ndarray, uv: np.ndarray, color: tuple, radius: int = 2) -> np.ndarray:
    """Draw points on image."""
    out = img.copy()
    H, W = out.shape[:2]
    for i in range(len(uv)):
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (u, v), radius, color, -1, lineType=cv2.LINE_AA)
    return out



def lidar_callback(meas: "carla.LidarMeasurement", buf: SensorBuffer):
    """Process LiDAR measurement."""
    arr = np.frombuffer(meas.raw_data, dtype=np.float32)
    if arr.size == 0:
        buf.lidar_pts_sensor = np.empty((0, 3), dtype=np.float32)
        return
    arr = arr.reshape((-1, 4))
    buf.lidar_pts_sensor = arr[:, :3].astype(np.float32)


def radar_callback(meas: "carla.RadarMeasurement", buf: SensorBuffer):
    """Process Radar measurement."""
    rows = []
    for d in meas:
        rows.append((float(d.depth), float(d.azimuth), float(d.altitude), float(d.velocity)))
    buf.radar_pts_sensor = np.array(rows, dtype=np.float32) if rows else np.empty((0, 4), dtype=np.float32)


def camera_callback(image: "carla.Image", buf: SensorBuffer):
    """Process camera image."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]
    rgb = bgr[:, :, ::-1].copy()
    buf.rgb_image = rgb



def save_spherical_frame(
    save_lidar_dir: str,
    save_radar_dir: str,
    save_camera_dir: str,
    t_now: float,
    lidar_cloud_ego: np.ndarray,
    radar_pts_ego: np.ndarray,
    radar_doppler: Optional[np.ndarray],
    rgb_image: np.ndarray,
    ego_pos: np.ndarray,
    ego_yaw: float,
    last_ego_pos: Optional[np.ndarray],
    last_time: Optional[float]
):
    """Save LiDAR/Radar in spherical coordinates + Doppler, and RGB image."""
    if last_ego_pos is not None and last_time is not None:
        dt = max(t_now - last_time, 1e-6)
        ego_vel_world = (ego_pos - last_ego_pos) / dt
    else:
        ego_vel_world = np.zeros(3, np.float32)

    stamp = ts_name(t_now)
    fname = f"{stamp}.npy"

    #lidar
    if lidar_cloud_ego is not None and lidar_cloud_ego.size:
        sph_lidar = cartesian_to_spherical_xyz(lidar_cloud_ego)
        doppler_lidar = np.zeros((sph_lidar.shape[0], 1), np.float32)
        lidar_out = np.concatenate([sph_lidar.astype(np.float32), doppler_lidar], axis=1)
        np.save(os.path.join(save_lidar_dir, fname), lidar_out)

    #radar
    if radar_pts_ego is not None and radar_pts_ego.size:
        sph_radar = cartesian_to_spherical_xyz(radar_pts_ego)
        if radar_doppler is None:
            R_e2w = R_ego_to_world(ego_yaw)
            radar_pts_world = (R_e2w @ radar_pts_ego.T).T
            norms = np.linalg.norm(radar_pts_world, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            u_r = radar_pts_world / norms
            v_rel_world = -ego_vel_world[None, :]
            doppler_vals = np.sum(v_rel_world * u_r, axis=1)
        else:
            doppler_vals = np.asarray(radar_doppler, dtype=np.float32)
            if len(doppler_vals) != len(radar_pts_ego):
                raise ValueError("Radar Doppler count must match radar point count")
        doppler_radar = doppler_vals.astype(np.float32)[:, None]
        radar_out = np.concatenate([sph_radar.astype(np.float32), doppler_radar], axis=1)
        np.save(os.path.join(save_radar_dir, fname), radar_out)

    # Save RGB image
    if rgb_image is not None:
        img_fname = f"{stamp}.png"
        cv2.imwrite(os.path.join(save_camera_dir, img_fname), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))



def set_sync(world: "carla.World", fps: float):
    """Enable synchronous mode."""
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    settings.no_rendering_mode = False
    world.apply_settings(settings)


def restore_settings(world: "carla.World", old_settings):
    """Restore original world settings."""
    world.apply_settings(old_settings)


def spawn_ego_vehicle(world: "carla.World", bp_lib: "carla.BlueprintLibrary", spawn_point_index: int = 0):
    """Spawn the ego vehicle at a spawn point."""
    ego_bp = bp_lib.find(EGO_VEHICLE_BP)
    ego_bp.set_attribute("role_name", EGO_ROLE_NAME)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available on this map.")

    if spawn_point_index >= len(spawn_points):
        spawn_point_index = random.randint(0, len(spawn_points) - 1)
    
    spawn_point = spawn_points[spawn_point_index]
    ego = world.try_spawn_actor(ego_bp, spawn_point)
    
    if ego is None:
        for _ in range(10):
            idx = random.randint(0, len(spawn_points) - 1)
            ego = world.try_spawn_actor(ego_bp, spawn_points[idx])
            if ego is not None:
                break
    
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")
    
    return ego


def spawn_traffic(world: "carla.World", bp_lib: "carla.BlueprintLibrary", num_vehicles: int, num_pedestrians: int):
    """Spawn traffic vehicles and pedestrians."""
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    vehicles = []
    pedestrians = []
    
    # Spawn vehicles
    vehicle_bps = bp_lib.filter("vehicle.*")
    for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle is not None:
            vehicles.append(vehicle)
    
    # Spawn pedestrians
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walker_controller_bp = bp_lib.find('controller.ai.walker')
    
    for _ in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            
            walker = world.try_spawn_actor(bp, spawn_point)
            if walker is not None:
                pedestrians.append(walker)
                walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                world.tick()
                walker_controller.start()
                walker_controller.go_to_location(world.get_random_location_from_navigation())
                walker_controller.set_max_speed(1.4)
    
    print(f" Spawned {len(vehicles)} vehicles and {len(pedestrians)} pedestrians")
    return vehicles + pedestrians



def run_collection(
    out_dir: str,
    num_frames: int,
    record: bool,
    visualize: bool,
    viz_every: int,
    spawn_index: int,
    physics_config: Optional[RadarPhysicsConfig] = None,
    town_name: str = TOWN_NAME,
):
    """Main data collection function."""
    
    if not CARLA_AVAILABLE:
        raise RuntimeError("CARLA Python API not found. Activate your CARLA PythonAPI .egg env first.")
    
    os.makedirs(out_dir, exist_ok=True)
    
    if record:
        save_root_spher = os.path.join(out_dir, "radelf_spherical")
        save_lidar_dir = os.path.join(save_root_spher, "lidar_spher")
        save_radar_dir = os.path.join(save_root_spher, "radar_spher")
        save_radar_4d_dir = os.path.join(save_root_spher, "radar_4d_spher")
        save_camera_dir = os.path.join(save_root_spher, "camera_rgb")
        os.makedirs(save_lidar_dir, exist_ok=True)
        os.makedirs(save_radar_dir, exist_ok=True)
        if physics_config is not None:
            os.makedirs(save_radar_4d_dir, exist_ok=True)
        os.makedirs(save_camera_dir, exist_ok=True)
        print(f" Recording enabled:")
        print(f"  LiDAR  {save_lidar_dir}")
        print(f"  Radar  {save_radar_dir}")
        if physics_config is not None:
            print(f"  4D radar {save_radar_4d_dir}")
        print(f"  Camera  {save_camera_dir}")
    
    if visualize:
        overlay_dir = os.path.join(out_dir, "overlay_frames")
        os.makedirs(overlay_dir, exist_ok=True)
        print(f" Overlay visualization {overlay_dir}")
    
    coverage_log_path = os.path.join(out_dir, "coverage_diagnostics.txt") #purely for testing
    coverage_log = None
    if DIAGNOSTIC_COVERAGE_CHECK and record:
        coverage_log = open(coverage_log_path, 'w')
        coverage_log.write("Frame,TotalPoints,AzMin,AzMax,AzRange,Front,Right,Back,Left\n")
        print(f" Coverage diagnostics {coverage_log_path}")

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    radar_physics_rng = (
        np.random.default_rng(physics_config.seed)
        if physics_config is not None else None
    )
    
    print(f" Loading world: {town_name}")
    world = client.load_world(town_name)
    bp_lib = world.get_blueprint_library()
    old_settings = world.get_settings()

    actors = []
    sensors = []

    try:
        set_sync(world, FPS)
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        print(f" Spawning ego vehicle...")
        ego = spawn_ego_vehicle(world, bp_lib, spawn_index)
        actors.append(ego)
        
        if USE_AUTOPILOT:
            ego.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(ego, 0)
            traffic_manager.distance_to_leading_vehicle(ego, 2.5)
            traffic_manager.vehicle_percentage_speed_difference(ego, -20)
            print(f" Autopilot enabled")
        
        print(f" Spawning traffic...")
        traffic_actors = spawn_traffic(world, bp_lib, NUM_VEHICLES, NUM_PEDESTRIANS)
        actors.extend(traffic_actors)
        
        for actor in traffic_actors:
            if 'vehicle' in actor.type_id:
                actor.set_autopilot(True, traffic_manager.get_port())

        print(f" Spawning sensors...")
        
        #needed for 3d lidar 
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROT_HZ))
        lidar_bp.set_attribute("points_per_second", str(LIDAR_PPS))
        lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("upper_fov", str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute("lower_fov", str(LIDAR_LOWER_FOV))
        if lidar_bp.has_attribute("horizontal_fov"):
            lidar_bp.set_attribute("horizontal_fov", str(LIDAR_HORIZONTAL_FOV))
        
        print(f"   LiDAR: {LIDAR_CHANNELS}ch, {LIDAR_ROT_HZ}Hz rotation, {LIDAR_HORIZONTAL_FOV}° horizontal FOV")

        # Radar
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(RADAR_HORIZONTAL_FOV))
        radar_bp.set_attribute("vertical_fov", str(RADAR_VERTICAL_FOV))
        radar_bp.set_attribute("range", str(RADAR_RANGE))
        radar_bp.set_attribute("points_per_second", str(RADAR_PPS))

        # Camera
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_W))
        cam_bp.set_attribute("image_size_y", str(CAM_H))
        cam_bp.set_attribute("fov", str(CAM_FOV))

        buf = SensorBuffer()

        lidar_tf = carla.Transform(
            carla.Location(x=LIDAR_TRANSFORM_CONFIG["x"], y=LIDAR_TRANSFORM_CONFIG["y"], z=LIDAR_TRANSFORM_CONFIG["z"]),
            carla.Rotation(pitch=LIDAR_TRANSFORM_CONFIG["pitch"], yaw=LIDAR_TRANSFORM_CONFIG["yaw"], roll=LIDAR_TRANSFORM_CONFIG["roll"]),
        )
        radar_tf = carla.Transform(
            carla.Location(x=RADAR_TRANSFORM_CONFIG["x"], y=RADAR_TRANSFORM_CONFIG["y"], z=RADAR_TRANSFORM_CONFIG["z"]),
            carla.Rotation(pitch=RADAR_TRANSFORM_CONFIG["pitch"], yaw=RADAR_TRANSFORM_CONFIG["yaw"], roll=RADAR_TRANSFORM_CONFIG["roll"]),
        )
        cam_tf = carla.Transform(
            carla.Location(x=CAM_TRANSFORM_CONFIG["x"], y=CAM_TRANSFORM_CONFIG["y"], z=CAM_TRANSFORM_CONFIG["z"]),
            carla.Rotation(pitch=CAM_TRANSFORM_CONFIG["pitch"], yaw=CAM_TRANSFORM_CONFIG["yaw"], roll=CAM_TRANSFORM_CONFIG["roll"]),
        )

        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=ego)
        radar = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

        sensors += [lidar, radar, cam]
        actors += [lidar, radar, cam]

        lidar.listen(lambda m: lidar_callback(m, buf))
        radar.listen(lambda m: radar_callback(m, buf))
        cam.listen(lambda im: camera_callback(im, buf))

        K = build_camera_K(CAM_W, CAM_H, CAM_FOV)

        print(f" Warming up ({WARMUP_FRAMES} frames)...")
        for _ in range(WARMUP_FRAMES):
            world.tick()

        print(f" Starting data collection ({num_frames} frames)...")
        
        last_ego_pos = None
        last_time = None
        frames_recorded = 0
        initial_pos = None
        movement_verified = False
        
        for frame in range(num_frames):
            world.tick()
            
            t_now = frame * DT
            ego_tf_now = ego.get_transform()
            ego_loc = ego_tf_now.location
            ego_pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z], dtype=np.float32)
            ego_yaw = ego_tf_now.rotation.yaw

            #take this out, purely for testing 
            
            if initial_pos is None:
                initial_pos = ego_pos.copy()
            # test car is actually moving for data collection
            if not movement_verified and frame > 20:
                distance_moved = np.linalg.norm(ego_pos - initial_pos)
                if distance_moved > 1.0:
                    movement_verified = True
                    print(f" Movement verified! Distance: {distance_moved:.2f}m")
                elif frame == 50:
                    print(f"  WARNING: Vehicle hasn't moved! Switching to manual control...")
                    ego.set_autopilot(False)
            
            if not USE_AUTOPILOT or (not movement_verified and frame > 50):
                control = carla.VehicleControl()
                control.throttle = MANUAL_DRIVE_THROTTLE
                control.steer = 0.0
                control.brake = MANUAL_DRIVE_BRAKE
                ego.apply_control(control)

            rgb_img = buf.rgb_image
            if rgb_img is None:
                continue

            lidar_sensor = buf.lidar_pts_sensor if buf.lidar_pts_sensor is not None else np.empty((0, 3), dtype=np.float32)
            radar_raw = buf.radar_pts_sensor if buf.radar_pts_sensor is not None else np.empty((0, 4), dtype=np.float32)
            radar_raw_4d = (
                apply_radar_physics(radar_raw, physics_config, radar_physics_rng)
                if physics_config is not None else radar_raw
            )

            lidar_world = transform_points_sensor_to_world(lidar.get_transform(), lidar_sensor)
            lidar_ego = world_points_to_ego(ego_tf_now, lidar_world)

            radar_xyz_sensor = radar_spherical_to_xyz(radar_raw[:, :3]) if len(radar_raw) else np.empty((0, 3), dtype=np.float32)
            radar_world = transform_points_sensor_to_world(radar.get_transform(), radar_xyz_sensor)
            radar_ego = world_points_to_ego(ego_tf_now, radar_world)
            radar_xyz_sensor_4d = (
                radar_spherical_to_xyz(radar_raw_4d[:, :3])
                if len(radar_raw_4d) else np.empty((0, 3), dtype=np.float32)
            )
            radar_world_4d = transform_points_sensor_to_world(
                radar.get_transform(), radar_xyz_sensor_4d
            )
            radar_ego_4d = world_points_to_ego(ego_tf_now, radar_world_4d)

            #test
            if DIAGNOSTIC_COVERAGE_CHECK and coverage_log is not None and len(lidar_ego) > 0:
                cov = check_azimuth_coverage(lidar_ego, frame)
                if cov is not None:
                    coverage_log.write(f"{cov['frame']},{cov['total_points']},"
                                     f"{cov['azimuth_min']:.1f},{cov['azimuth_max']:.1f},"
                                     f"{cov['azimuth_range']:.1f},"
                                     f"{cov['quadrants']['Front']},"
                                     f"{cov['quadrants']['Right']},"
                                     f"{cov['quadrants']['Back']},"
                                     f"{cov['quadrants']['Left']}\n")
                    
                    if frame > WARMUP_FRAMES and cov['azimuth_range'] < 300:
                        print(f"    WARNING Frame {frame}: Limited azimuth range ({cov['azimuth_range']:.0f}°)")

            if record:
                save_spherical_frame(
                    save_lidar_dir,
                    save_radar_dir,
                    save_camera_dir,
                    t_now,
                    lidar_ego,
                    radar_ego,
                    radar_raw[:, 3],
                    rgb_img,
                    ego_pos,
                    ego_yaw,
                    last_ego_pos,
                    last_time
                )
                if physics_config is not None:
                    save_spherical_frame(
                        save_lidar_dir,
                        save_radar_4d_dir,
                        save_camera_dir,
                        t_now,
                        np.empty((0, 3), dtype=np.float32),
                        radar_ego_4d,
                        radar_raw_4d[:, 3],
                        None,
                        ego_pos,
                        ego_yaw,
                        last_ego_pos,
                        last_time,
                    )
                frames_recorded += 1

            if visualize and (frame % viz_every == 0):
                lidar_vis = downsample_points(lidar_world, OVERLAY_DOWNSAMPLE_LIDAR)
                radar_vis = downsample_points(radar_world, OVERLAY_DOWNSAMPLE_RADAR)

                lidar_cam = world_to_camera_frame(cam.get_transform(), lidar_vis)
                radar_cam = world_to_camera_frame(cam.get_transform(), radar_vis)
                
                lidar_uv = project_camera_xyz_to_pixels(lidar_cam, K, CAM_W, CAM_H)
                radar_uv = project_camera_xyz_to_pixels(radar_cam, K, CAM_W, CAM_H)

                overlay_img = rgb_img.copy()
                if len(lidar_uv) > 0:
                    overlay_img = draw_points_on_image(overlay_img, lidar_uv, OVERLAY_LIDAR_COLOR, OVERLAY_POINT_RADIUS)
                if len(radar_uv) > 0:
                    overlay_img = draw_points_on_image(overlay_img, radar_uv, OVERLAY_RADAR_COLOR, OVERLAY_POINT_RADIUS + 1)

                cv2.putText(overlay_img, f"Frame: {frame}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(overlay_img, f"LiDAR pts: {len(lidar_ego)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay_img, f"Radar pts: {len(radar_ego)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                if DIAGNOSTIC_COVERAGE_CHECK and len(lidar_ego) > 0:
                    cov = check_azimuth_coverage(lidar_ego, frame)
                    if cov is not None:
                        cv2.putText(overlay_img, 
                                   f"Coverage: {cov['azimuth_range']:.0f}deg "
                                   f"F:{cov['quadrants']['Front']} R:{cov['quadrants']['Right']} "
                                   f"B:{cov['quadrants']['Back']} L:{cov['quadrants']['Left']}", 
                                   (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                overlay_path = os.path.join(overlay_dir, f"overlay_{frame:06d}.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

            if frame % DEBUG_PRINT_EVERY == 0:
                vel = ego.get_velocity()
                speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                distance_from_start = np.linalg.norm(ego_pos - initial_pos)
                print(f"[{frame:05d}/{num_frames}] Speed: {speed:.1f} km/h | "
                      f"LiDAR: {len(lidar_ego)} pts | Radar: {len(radar_ego)} pts | "
                      f"Dist: {distance_from_start:.1f}m")

            last_ego_pos = ego_pos.copy()
            last_time = t_now

        print(f"\n Data collection complete!")
        print(f"  Recorded {frames_recorded} frames")
        if initial_pos is not None:
            total_distance = np.linalg.norm(ego_pos - initial_pos)
            print(f"  Total distance: {total_distance:.2f}m")
        if record:
            print(f"  Data saved to: {save_root_spher}/")
        if coverage_log is not None:
            print(f"  Coverage log: {coverage_log_path}")
            print(f"\n  Run analysis: python {__file__} --analyze {out_dir}")

    finally:
        if coverage_log is not None:
            coverage_log.close()
        
        print("\n Cleaning up...")
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass

        destroyed = set()
        for a in reversed(actors):
            try:
                if a is None:
                    continue
                if a.id in destroyed:
                    continue
                a.destroy()
                destroyed.add(a.id)
            except Exception:
                pass

        try:
            restore_settings(world, old_settings)
        except Exception:
            pass


# look at outlier measurements 
def analyze_lidar_file(filepath):
    """Analyze a single LiDAR file for angular distribution."""
    data = np.load(filepath)
    
    if data.size == 0:
        return None
    
    ranges = data[:, 0]
    azimuths = data[:, 1]
    elevations = data[:, 2]
    
    azimuth_deg = np.degrees(azimuths)
    azimuth_deg = (azimuth_deg + 180) % 360 - 180
    
    return {
        'num_points': len(data),
        'azimuth_deg': azimuth_deg,
        'ranges': ranges,
        'elevations': elevations,
        'azimuth_rad': azimuths
    }


def plot_angular_distribution(results, output_dir):
    """Plot the angular distribution of LiDAR points."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    azimuth_all = np.concatenate([r['azimuth_deg'] for r in results if r is not None])
    ranges_all = np.concatenate([r['ranges'] for r in results if r is not None])
    
    # Histogram of azimuth angles
    axes[0, 0].hist(azimuth_all, bins=72, range=(-180, 180), edgecolor='black')
    axes[0, 0].set_xlabel('Azimuth (degrees)')
    axes[0, 0].set_ylabel('Point Count')
    axes[0, 0].set_title('Azimuth Distribution (Should be uniform for 360° coverage)')
    axes[0, 0].axvline(x=0, color='r', linestyle='--', label='Forward')
    axes[0, 0].axvline(x=90, color='g', linestyle='--', label='Right')
    axes[0, 0].axvline(x=-90, color='b', linestyle='--', label='Left')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Polar plot
    axes[0, 1].remove()
    axes[0, 1] = fig.add_subplot(2, 2, 2, projection='polar')
    
    sample_size = min(10000, len(azimuth_all))
    sample_idx = np.random.choice(len(azimuth_all), sample_size, replace=False)
    
    azimuth_sample = np.radians(azimuth_all[sample_idx])
    ranges_sample = ranges_all[sample_idx]
    
    axes[0, 1].scatter(azimuth_sample, ranges_sample, s=1, alpha=0.5)
    axes[0, 1].set_title('Polar View (Top-down)', pad=20)
    axes[0, 1].set_theta_zero_location('N')
    axes[0, 1].set_theta_direction(-1)
    
    # Range distribution
    axes[1, 0].hist(ranges_all, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Range (meters)')
    axes[1, 0].set_ylabel('Point Count')
    axes[1, 0].set_title('Range Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quadrant coverage
    quadrants = {
        'Front\n(45°)': (azimuth_all >= -45) & (azimuth_all <= 45),
        'Right\n(45-135°)': (azimuth_all > 45) & (azimuth_all <= 135),
        'Back\n(135-180°)': (azimuth_all > 135) | (azimuth_all < -135),
        'Left\n(135 to 45°)': (azimuth_all >= -135) & (azimuth_all < -45)
    }
    
    quadrant_counts = {name: np.sum(mask) for name, mask in quadrants.items()}
    
    axes[1, 1].bar(quadrant_counts.keys(), quadrant_counts.values())
    axes[1, 1].set_ylabel('Point Count')
    axes[1, 1].set_title('Points per Quadrant (Should be roughly equal)')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "coverage_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Plot saved to: {output_path}")
    plt.close()
    
    return quadrant_counts

""" 
lidar sensor debug 
remove eventually 

"""
def run_analysis(data_dir: str, max_files: int):
    """Analyze collected LiDAR data."""
    lidar_dir = os.path.join(data_dir, "radelf_spherical", "lidar_spher")
    
    if not os.path.exists(lidar_dir):
        print(f"ERROR: Directory not found: {lidar_dir}")
        return
    
    lidar_files = sorted(glob(os.path.join(lidar_dir, "*.npy")))
    
    if not lidar_files:
        print(f"ERROR: No .npy files found in {lidar_dir}")
        return
    
    print(f"Found {len(lidar_files)} LiDAR files")
    print(f"Analyzing up to {max_files} files...\n")
    
    results = []
    for i, filepath in enumerate(lidar_files[:max_files]):
        result = analyze_lidar_file(filepath)
        if result is not None:
            results.append(result)
            if i < 5:
                print(f"File {i+1}: {os.path.basename(filepath)}")
                print(f"  Points: {result['num_points']}")
                print(f"  Azimuth: [{result['azimuth_deg'].min():.1f}°, {result['azimuth_deg'].max():.1f}°]")
    
    if not results:
        print("ERROR: No valid LiDAR data found")
        return
    
    # Create analysis output directory
    analysis_dir = os.path.join(data_dir, "coverage_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"\nGenerating analysis plots...")
    quadrant_counts = plot_angular_distribution(results, analysis_dir)
    
    # lidar sesnor debug 
    report_path = os.path.join(analysis_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LIDAR COVERAGE ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        total_points = sum(r['num_points'] for r in results)
        f.write(f"Files analyzed: {len(results)}\n")
        f.write(f"Total points: {total_points:,}\n")
        f.write(f"Average points per frame: {total_points/len(results):.0f}\n\n")
        
        f.write("Quadrant Distribution:\n")
        for name, count in quadrant_counts.items():
            percentage = 100 * count / sum(quadrant_counts.values())
            f.write(f"  {name:25s}: {count:8,} ({percentage:5.1f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("DIAGNOSIS\n")
        f.write("="*60 + "\n\n")
        
        percentages = [100 * count / sum(quadrant_counts.values()) for count in quadrant_counts.values()]
        max_diff = max(percentages) - min(percentages)
        
        if max_diff > 30:
            f.write("  WARNING: Uneven distribution detected!\n")
            f.write("   LiDAR is NOT capturing 360° properly.\n")
            f.write("   Possible causes:\n")
            f.write("   - LiDAR rotation frequency too low\n")
            f.write("   - Coordinate transformation error\n")
            f.write("   - Sensor configuration issue\n")
        elif max_diff > 15:
            f.write("  NOTICE: Slightly uneven distribution\n")
            f.write("   This could be normal due to environment.\n")
        else:
            f.write(" Distribution looks good!\n")
            f.write("  LiDAR appears to be capturing 360° coverage properly.\n")
    
    print(f"✓ Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    total_points = sum(r['num_points'] for r in results)
    print(f"Files analyzed: {len(results)}")
    print(f"Total points: {total_points:,}")
    print(f"Average points per frame: {total_points/len(results):.0f}")
    
    print("\nQuadrant Distribution:")
    for name, count in quadrant_counts.items():
        percentage = 100 * count / sum(quadrant_counts.values())
        print(f"  {name:25s}: {count:8,} ({percentage:5.1f}%)")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    percentages = [100 * count / sum(quadrant_counts.values()) for count in quadrant_counts.values()]
    max_diff = max(percentages) - min(percentages)
    
    if max_diff > 30:
        print("  WARNING: Uneven distribution!")
        print("   LiDAR is NOT capturing 360° properly.")
        print("   Check the analysis plots for details.")
    elif max_diff > 15:
        print(" NOTICE: Slightly uneven distribution")
        print("   This could be normal due to environment.")
    else:
        print(" Distribution looks good!")
        print("  LiDAR appears to be capturing 360° properly.")
    
    print("="*60)
    print(f"\nAnalysis complete! Results saved to: {analysis_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="CARLA Data Collection with Built-in Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data
  python %(prog)s --record --frames 2000 --out-dir ./training_data
  python %(prog)s --record --visualize --frames 2000
  
  # Analyze collected data
  python %(prog)s --analyze ./training_data --max-files 50
        """
    )
    
    # Mode selection
    parser.add_argument("--analyze", type=str, metavar="DIR", 
                       help="Analyze collected data in DIR (skips collection)")
    
    # Collection options
    parser.add_argument("--out-dir", type=str, default="training_data", 
                       help="Output directory (default: training_data)")
    parser.add_argument("--frames", type=int, default=2000, 
                       help="Number of frames to record (default: 2000)")
    parser.add_argument("--record", action="store_true", 
                       help="Enable frame recording")
    parser.add_argument("--visualize", action="store_true", 
                       help="Save overlay visualizations")
    parser.add_argument("--viz-every", type=int, default=30, 
                       help="Save visualization every N frames (default: 30)")
    parser.add_argument("--spawn-index", type=int, default=0, 
                       help="Spawn point index for ego vehicle (default: 0)")
    parser.add_argument(
        "--town",
        type=str,
        default=TOWN_NAME,
        help=f"CARLA map to load (default: {TOWN_NAME})",
    )
    parser.add_argument(
        "--physics-config",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON file of RadarPhysicsConfig fields. Native radar remains in "
            "radar_spher; modified output is written to radar_4d_spher."
        ),
    )
    
    # Analysis options
    parser.add_argument("--max-files", type=int, default=50, 
                       help="Max files to analyze (default: 50)")
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analysis mode
        print("="*60)
        print("LIDAR COVERAGE ANALYSIS MODE")
        print("="*60)
        run_analysis(args.analyze, args.max_files)
    else:
        physics_config = None
        if args.physics_config:
            with open(args.physics_config, encoding="utf-8") as config_file:
                config_values = json.load(config_file)
            config_values["enabled"] = True
            physics_config = RadarPhysicsConfig(**config_values)

        # Collection mode
        if not args.record and not args.visualize:
            print("  WARNING: Neither --record nor --visualize enabled.")
            print("   No output will be saved!")
            print("   Use --record to save data, --visualize for overlays, or both.\n")
        
        print("="*60)
        print("CARLA DATA COLLECTION MODE")
        print("="*60)
        run_collection(
            out_dir=args.out_dir,
            num_frames=args.frames,
            record=args.record,
            visualize=args.visualize,
            viz_every=args.viz_every,
            spawn_index=args.spawn_index,
            physics_config=physics_config,
            town_name=args.town,
        )


if __name__ == "__main__":
    main()

#python3 new_delft2.py --record --visualize --viz-every 30 --frames 2000