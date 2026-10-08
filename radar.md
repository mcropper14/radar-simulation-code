## Radar Measurement Model

The CARLA radar callback records each detection as
`[range_m, azimuth_rad, elevation_rad, velocity_mps]`. The native stream is
written to `radar_spher`. When `--physics-config` is supplied, the same native
detections are passed through the optional measurement model and the result is
written to `radar_4d_spher`.

### Noise and Detection Loss

The model in `radar_pipeline/radar_physics.py` applies the following steps to
each native detection:

1. A detection is retained with a range-dependent probability. The probability
	 is linearly interpolated between `detection_probability_at_zero_m` and
	 `detection_probability_at_max_range` over `detection_probability_range_m`.
2. Independent zero-mean Gaussian noise is added to range, azimuth, elevation,
	 and radial velocity. The standard deviations are configured by
	 `range_noise_std_m`, `azimuth_noise_std_rad`, `elevation_noise_std_rad`,
	 and `doppler_noise_std_mps`.
3. Detections whose perturbed range is no longer positive are removed.

Thus, for example, the perturbed range is

$$r' = r + \epsilon_r, \qquad \epsilon_r \sim \mathcal{N}(0,\sigma_r^2).$$

The random generator can be made repeatable with the configuration `seed`.
Setting all standard deviations to zero and the detection probabilities to one
leaves the native measurements unchanged (apart from the configured clutter or
multipath effects).

### Multipath Reflection and Clutter

The implemented multipath model is a configurable ghost-detection model, not a
full electromagnetic ray-tracing simulation. For every post-noise detection,
the code draws a Bernoulli trial with probability `multipath_probability`.
For each selected detection it creates a copy and adds a uniformly sampled
extra path length:

$$r_{\mathrm{ghost}} = r' + U(d_{\min}, d_{\max}),$$

where `d_min` and `d_max` are
`multipath_extra_range_min_m` and `multipath_extra_range_max_m`. The ghost
keeps the original radial velocity and receives independent angular jitter
using the azimuth and elevation noise standard deviations. The original target
detection is retained, so multipath increases the number of false/duplicate
returns.

Clutter is generated separately. Its count is sampled as
`Poisson(clutter_per_detection * number_of_detections)`. Each clutter point gets
a random range and angle inside the configured bounds and a zero-mean Gaussian
velocity with standard deviation `clutter_velocity_std_mps`.

### How Moving-Object Velocity Is Calculated

CARLA supplies `d.velocity` for each radar return. This is the relative radial
velocity between the detected object and the radar sensor, and it already
includes both target motion and ego-sensor motion. The project uses the
convention **positive = approaching**. It is not the object speed or a full
three-dimensional velocity vector.

Conceptually, let

$$\mathbf{u} = \frac{\mathbf{p}_{target}-\mathbf{p}_{sensor}}
											{\|\mathbf{p}_{target}-\mathbf{p}_{sensor}\|}$$

be the unit line-of-sight vector. With world-frame target and sensor
velocities $\mathbf{v}_{target}$ and $\mathbf{v}_{sensor}$, the stored
approaching radial velocity is

$$v_r = (\mathbf{v}_{sensor}-\mathbf{v}_{target})\cdot\mathbf{u}.$$

Therefore, motion toward the sensor is positive, motion away is negative, and
sideways motion contributes little or nothing. The helper
`approaching_radial_velocity()` in `radar_pipeline/radar_physics.py` implements
this projection when an explicit recomputation is needed.

During closed-loop decoding, the Doppler bin is converted back to velocity by

$$v_r = (k - D/2)\,\Delta v,$$

where $k$ is the selected Doppler-bin index, $D$ is `vel_fft_size`, and
`\Delta v` is `vel_bin_size`. The decoder selects $k$ using the maximum radar
power in the Doppler dimension for each occupied range/azimuth cell.

If a frame has no native Doppler values, the dataset writer has a fallback that
uses ego motion only: it projects $-\mathbf{v}_{ego}$ onto each radar ray. That
fallback estimates the apparent velocity caused by sensor motion; it cannot
recover the independent velocity of a moving target.

## Running With the Physics Model

Create a JSON file containing any `RadarPhysicsConfig` fields, for example:

```json
{
	"range_noise_std_m": 0.05,
	"azimuth_noise_std_rad": 0.005,
	"elevation_noise_std_rad": 0.005,
	"doppler_noise_std_mps": 0.25,
	"detection_probability_at_zero_m": 0.98,
	"detection_probability_at_max_range": 0.65,
	"detection_probability_range_m": 100.0,
	"clutter_per_detection": 0.05,
	"multipath_probability": 0.02,
	"seed": 7
}
```

Pass it to the collector with:

```bash
python3 radar_pipeline/new_delft2.py --record --frames 2000 \
	--out-dir ./test_lidar_images2_4d --physics-config ./radar_physics.json
```
