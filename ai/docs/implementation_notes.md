# Implementation notes - Navigation2DEnv (Week 9)

This document records the design decisions of the simplified 2D simulator. It is a numerical
simulator written in Python/NumPy; it is **not** Gazebo and has not been validated against the
physical robot. All parameters are proposals stored in `configs/environment.yaml`.

## 1. Module map

| Module | Responsibility |
|---|---|
| `config.py` | Typed, validated `EnvConfig` / `RewardConfig` loaded from YAML |
| `geometry.py` | Angle wrapping, Euler unicycle step, goal distance/bearing, swept-disc collision test |
| `sensors.py` | 8-ray range sensor (analytic ray-circle and ray-box intersection) |
| `rewards.py` | Reward function and its logged components |
| `scenarios.py` | Seeded generation of `open` / `indoor` / `narrow` scenarios, fixed validation/test sets |
| `planning.py` | Grid navigability check (flood fill) and approximate shortest-path reference for SPL |
| `navigation_env.py` | `Navigation2DEnv(gym.Env)` - spaces, reset/step, outcomes, episode statistics |
| `baselines.py` | Random policy and an observation-only goal-seeking heuristic |
| `evaluation.py` | Episode runner on fixed scenario lists, metrics, Wilson confidence interval |
| `visualization.py` | Static scenario / trajectory plots, `render()` RGB array |

## 2. Robot model

State `(x, y, theta)` in the world frame (x right, y up, origin at the lower-left corner,
theta counter-clockwise from +x, wrapped to [-pi, pi)). One control step (dt = 0.2 s):

```
x'     = x + v cos(theta) dt
y'     = y + v sin(theta) dt
theta' = wrap(theta + omega dt)
```

This is exactly the forward-Euler model of the Week 9 specification (`geometry.unicycle_step`).
Within one step the robot therefore translates along a straight segment with the *old* heading,
then rotates. Because the per-step displacement is at most 0.35 * 0.2 = 0.07 m, the
discretisation error relative to the exact arc is small, but it is a modelling simplification.

Action mapping (`Navigation2DEnv.action_to_velocity`): `v = (u_v + 1)/2 * 0.35`,
`omega = u_w * 1.5`. Actions are validated (shape (2,), finite) and clipped to [-1, 1].
**Reverse motion is disabled** in this experiment (v >= 0).

## 3. Collision geometry

* The robot is a disc of radius 0.18 m. Obstacles are discs; the world is the square [0, 6]^2.
* Collision is tested in configuration space: each obstacle is inflated by the robot radius,
  the walls are moved inward by the robot radius, and the robot centre is swept along the step
  segment. `first_contact_fraction` solves the segment-circle quadratic analytically, so a
  collision is detected even if a single step jumped completely across a thin obstacle
  (no tunnelling; see `test_swept_contact_obstacle_and_no_tunnelling`).
* On collision the robot is placed at the first contact point (no penetration) and the episode
  terminates.

## 4. Range sensor

8 rays at heading + k * 45 deg (k = 0..7, 360 deg field of view, ray 0 straight ahead, counter-
clockwise). Each reading is the analytic distance from the robot centre to the first circle or
wall, clipped to 3.5 m and divided by 3.5. The readings are geometric, not fabricated: unit
tests compare them against hand-computed distances.

## 5. Observation and action spaces

`Box(low=[0]*9 + [-1, -1], high=[1]*11, dtype=float32)`:
`[l_1..l_8, d_goal / (6*sqrt(2)), sin(alpha), cos(alpha)]`, where alpha is the goal bearing
relative to the heading. Obstacle coordinates, the goal position in world coordinates and the
robot pose are **not** part of the observation; they appear only in `info` for diagnostics.

`Box(-1, 1, (2,), float32)` for `(u_v, u_w)`.

## 6. Reward

```
r_t = k_p (d_t - d_{t+1}) + k_g I_goal - k_c I_collision - k_t
k_p = 10, k_g = 20, k_c = 20, k_t = 0.05
```

Rationale (initial choice, not claimed optimal):

* The progress term is a difference of the potential `-k_p d`, i.e. potential-based shaping
  (Ng et al., 1999). At maximum speed it yields at most 0.7 per step, and summed over an
  episode it equals `k_p (d_0 - d_T)` regardless of the route, so it cannot be farmed by
  oscillating.
* A typical start-goal distance of 3-5 m gives 30-50 progress reward; `k_g = 20` makes reaching
  the goal clearly better than stopping just outside the tolerance, and `k_c = 20` makes a
  collision cost roughly as much as half of a typical episode's progress.
* `k_t = 0.05` (12.5 over a full 250-step timeout) discourages dawdling without dominating.
* Terminal terms are paid on the terminal step only; `step()` raises if called after
  termination, so a bonus cannot be collected twice. All terms are bounded, so the reward is
  finite.

## 7. Episodes and outcomes

| Outcome | Condition | Gymnasium flag |
|---|---|---|
| `success` | distance to goal <= 0.25 m after the step (and no collision) | `terminated=True` |
| `collision` | swept disc touches an obstacle or wall during the step | `terminated=True` |
| `timeout` | 250 steps without success/collision | `truncated=True` |

The terminal `info` contains `outcome`, `is_success`, `is_collision`, `is_timeout` and
`episode_summary` (return, length, path length, initial/final goal distance, time to goal,
trajectory, scenario id).

## 8. Scenarios

| Category | Content |
|---|---|
| `open` | 1-3 circles, radius 0.20-0.40 m |
| `indoor` | 5-8 circles, radius 0.20-0.50 m |
| `narrow` | a straight partition (posts of radius 0.15 m every 0.2 m) across the room with one doorway (>= 0.60-0.80 m surface gap), plus 1-3 extra circles; start and goal on opposite sides; randomly vertical or horizontal |

Start: clearance >= robot radius + 0.15 m. Goal: clearance >= robot radius + 0.05 m. Start-goal
distance >= 2 m. A scenario is accepted only if a grid flood fill in configuration space
(robot radius + 0.03 m inflation, 0.05 m cells) connects start and goal. Generation is fully
determined by `(category, seed)`.

Seed ranges are disjoint: training resets draw seeds in [0, 10^6) from `env.np_random`,
validation uses 1 000 000 + i (30 scenarios, used only for checkpoint monitoring), and the
held-out test set uses 2 000 000 + i (50 per category = 150 scenarios).

## 9. Approximate SPL

SPL (Anderson et al., 2018) requires the shortest collision-free path length. `planning.py`
computes an 8-connected Dijkstra path on the inflated grid, shortcuts it with exact swept-disc
line-of-sight checks, and uses its length `L`. This is an approximation of the true continuous
shortest path (usually slightly longer), so the metric is reported as **SPL_approx** and should
be interpreted as indicative, not exact.

## 10. Known simplifications

* No sensor noise, latency, wheel slip, acceleration limits or actuator dynamics.
* Obstacles are static circles only; walls in `narrow` are chains of circles.
* Euler integration with straight-segment motion per step.
* The robot always starts at rest; there is no velocity in the observation.
