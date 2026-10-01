"""Physics-based perturbations and Doppler helpers for CARLA radar data.

CARLA RadarDetection.velocity is the relative radial velocity between the
target and the radar sensor.  The convention used here is positive when the
target is approaching the sensor, matching the existing ACC code.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RadarPhysicsConfig:
    """Configurable measurement effects applied after CARLA radar sampling."""

    enabled: bool = False
    range_noise_std_m: float = 0.0
    azimuth_noise_std_rad: float = 0.0
    elevation_noise_std_rad: float = 0.0
    doppler_noise_std_mps: float = 0.0
    detection_probability_at_zero_m: float = 1.0
    detection_probability_at_max_range: float = 1.0
    detection_probability_range_m: float = 100.0
    clutter_per_detection: float = 0.0
    clutter_min_range_m: float = 2.0
    clutter_max_range_m: float = 100.0
    clutter_horizontal_fov_deg: float = 70.0
    clutter_vertical_fov_deg: float = 20.0
    clutter_velocity_std_mps: float = 2.0
    multipath_probability: float = 0.0
    multipath_extra_range_min_m: float = 2.0
    multipath_extra_range_max_m: float = 15.0
    seed: Optional[int] = None

    def validate(self) -> None:
        if not 0.0 <= self.detection_probability_at_zero_m <= 1.0:
            raise ValueError("detection_probability_at_zero_m must be in [0, 1]")
        if not 0.0 <= self.detection_probability_at_max_range <= 1.0:
            raise ValueError("detection_probability_at_max_range must be in [0, 1]")
        if self.detection_probability_range_m <= 0:
            raise ValueError("detection_probability_range_m must be positive")
        if self.clutter_per_detection < 0:
            raise ValueError("clutter_per_detection must be non-negative")
        if not 0.0 <= self.multipath_probability <= 1.0:
            raise ValueError("multipath_probability must be in [0, 1]")
        if self.multipath_extra_range_min_m < 0:
            raise ValueError("multipath_extra_range_min_m must be non-negative")
        if self.multipath_extra_range_max_m < self.multipath_extra_range_min_m:
            raise ValueError("multipath extra range bounds are invalid")


def approaching_radial_velocity(
    sensor_velocity_world: np.ndarray,
    target_velocity_world: np.ndarray,
    target_position_world: np.ndarray,
    sensor_position_world: np.ndarray,
) -> float:
    """Compute positive-toward-sensor radial velocity in m/s.

    ``u`` points from sensor to target.  Relative velocity of the target with
    respect to the sensor is ``target_velocity - sensor_velocity``.  Negating
    its projection onto ``u`` makes approaching motion positive.
    """
    sensor_velocity_world = np.asarray(sensor_velocity_world, dtype=np.float64)
    target_velocity_world = np.asarray(target_velocity_world, dtype=np.float64)
    target_position_world = np.asarray(target_position_world, dtype=np.float64)
    sensor_position_world = np.asarray(sensor_position_world, dtype=np.float64)
    ray = target_position_world - sensor_position_world
    distance = np.linalg.norm(ray)
    if distance <= 1e-9:
        return 0.0
    unit_ray = ray / distance
    return float(np.dot(sensor_velocity_world - target_velocity_world, unit_ray))


def _range_detection_probability(ranges: np.ndarray, config: RadarPhysicsConfig):
    fraction = np.clip(
        ranges / config.detection_probability_range_m, 0.0, 1.0
    )
    return (
        config.detection_probability_at_zero_m
        + fraction
        * (
            config.detection_probability_at_max_range
            - config.detection_probability_at_zero_m
        )
    )


def apply_radar_physics(
    radar_raw: np.ndarray,
    config: RadarPhysicsConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply detection loss, measurement noise, multipath, and clutter.

    Input/output rows are ``[range_m, azimuth_rad, elevation_rad,
    approaching_velocity_mps]``.  The function never mutates the input.
    """
    config.validate()
    points = np.asarray(radar_raw, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected radar data with shape (N, 4+), got {points.shape}")
    points = points[:, :4].copy()
    if not config.enabled:
        return points
    rng = rng or np.random.default_rng(config.seed)

    probabilities = _range_detection_probability(points[:, 0], config)
    keep = rng.random(len(points)) <= probabilities
    points = points[keep]
    if points.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    points[:, 0] += rng.normal(0.0, config.range_noise_std_m, len(points))
    points[:, 1] += rng.normal(0.0, config.azimuth_noise_std_rad, len(points))
    points[:, 2] += rng.normal(0.0, config.elevation_noise_std_rad, len(points))
    points[:, 3] += rng.normal(0.0, config.doppler_noise_std_mps, len(points))
    points = points[points[:, 0] > 0.0]

    if config.multipath_probability > 0.0 and len(points):
        multipath_keep = rng.random(len(points)) < config.multipath_probability
        ghosts = points[multipath_keep].copy()
        if len(ghosts):
            ghosts[:, 0] += rng.uniform(
                config.multipath_extra_range_min_m,
                config.multipath_extra_range_max_m,
                len(ghosts),
            )
            ghosts[:, 1] += rng.normal(0.0, config.azimuth_noise_std_rad, len(ghosts))
            ghosts[:, 2] += rng.normal(0.0, config.elevation_noise_std_rad, len(ghosts))
            points = np.vstack((points, ghosts))

    clutter_count = rng.poisson(config.clutter_per_detection * len(points))
    if clutter_count:
        clutter = np.column_stack((
            rng.uniform(config.clutter_min_range_m, config.clutter_max_range_m, clutter_count),
            rng.uniform(-np.deg2rad(config.clutter_horizontal_fov_deg) / 2,
                        np.deg2rad(config.clutter_horizontal_fov_deg) / 2, clutter_count),
            rng.uniform(-np.deg2rad(config.clutter_vertical_fov_deg) / 2,
                        np.deg2rad(config.clutter_vertical_fov_deg) / 2, clutter_count),
            rng.normal(0.0, config.clutter_velocity_std_mps, clutter_count),
        )).astype(np.float32)
        points = np.vstack((points, clutter))

    return points.astype(np.float32, copy=False)
