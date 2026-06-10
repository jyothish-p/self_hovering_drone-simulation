# Environment Configuration

## RLEnvironment CONFIG

| Parameter | Value | Description |
|-----------|-------|-------------|
| targetX/Y/Z | 0, 10, 0 | Hover target position (10 m altitude) |
| hoverThrust | ~3.43 N | Thrust per motor at equilibrium (auto-calibrated) |
| maxEpisodeSteps | 2000 | Max steps per RL episode |
| thrustDelta | 0.3 N | Per-action thrust increment |
| crashHeight | 0.4 m | Ground collision detection threshold |
| tiltPrimaryDeg | 2° | Tilt threshold for +50 primary reward |
| tiltGoodDeg | 5° | Tilt threshold for +20 good reward |
| tiltSevereDeg | 15° | Tilt threshold for −100 penalty |
| tiltExtremeDeg | 25° | Tilt threshold for −200 penalty |
| tiltResetDeg | 35° | Episode reset on extreme tilt |
| altPreciseLo/Hi | 9.8–10.2 m | ±0.2 m precision altitude band (+30) |
| altModLo/Hi | 9.5–10.5 m | ±0.5 m moderate altitude band (+10) |
| velStableThresh | 0.3 m/s | Vertical velocity stability threshold (+15) |
| angStableThresh | 0.3 rad/s | Angular velocity stability threshold (+10) |
| thrustSymmTol | 1.0 N | Symmetry tolerance for +20 reward |
| thrustAsymPen | 4.0 N | Asymmetry threshold for −25 penalty |

## DroneModel SPECS

| Parameter | Value |
|-----------|-------|
| Total mass | 1.4 kg |
| Motor count | 4 |
| Arm length | 225 mm |
| Max thrust/motor | 15 N |
| Motor time constant | 50 ms (first-order lag) |
| Inertia Ixx/Izz | 0.0277 kg·m² |
| Inertia Iyy | 0.0522 kg·m² |
| Gravity | 9.81 m/s² |

## Stage 1 PD Controller Tuning

| Gain | Value |
|------|-------|
| Altitude P | 0.6 (target climb rate = err × 0.6) |
| Altitude D (vy error) | 0.8 thrust/m·s⁻¹ |
| Roll/Pitch P | 2.8 thrust-N/rad |
| Roll/Pitch D | 0.45 thrust-N/rad·s⁻¹ |
| Stable threshold | 0.25 m altitude, 0.25 m/s vy, 4° tilt |
| Transition time | 60 frames (1 second) at stable hover |
