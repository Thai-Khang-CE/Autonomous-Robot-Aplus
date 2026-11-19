# ROS 2 / Gazebo integration contract (draft, NOT implemented)

Status on 2026-11-13: **outstanding dependency.** The machine used for Week 9 runs Windows 11
without an Ubuntu WSL distribution (only `docker-desktop` is registered); `ros2` and `gz` are
not on the PATH. No ROS 2 Jazzy / Gazebo Harmonic environment exists, so no policy has been run
in Gazebo. This document fixes the interface that a future ROS 2 node must satisfy so that a
policy trained in `Navigation2DEnv` can be executed (and later re-trained) in Gazebo.

Recommended platform when integration starts: Ubuntu 24.04 (native or WSL2) + ROS 2 Jazzy +
Gazebo Harmonic + `ros_gz` (the officially paired versions). Do not mix ROS 2 Jazzy with
Gazebo Classic or with non-matching Gazebo releases.

## 1. Inputs

| Quantity | ROS 2 source | Message type | Notes |
|---|---|---|---|
| Range scan | `/scan` | `sensor_msgs/msg/LaserScan` | from Gazebo `gpu_lidar` via `ros_gz_bridge` |
| Robot pose | `/odom` (sim) or `/amcl_pose` / TF `map->base_link` | `nav_msgs/msg/Odometry` / `geometry_msgs/msg/PoseWithCovarianceStamped` | needed only to compute goal distance and bearing |
| Goal | `/goal_pose` | `geometry_msgs/msg/PoseStamped` | in the `map` (or `odom`) frame |

## 2. Observation preprocessing (must match `Navigation2DEnv._observation`)

1. **Rays.** From the `LaserScan`, take 8 sectors centred at 0, 45, ..., 315 deg relative to
   `base_link` +x (counter-clockwise). For each sector use the **minimum** valid range within
   +/- 22.5 deg (more conservative than the single ray in simulation; document the change if
   used). Replace `inf`/`NaN`/`> range_max` by 3.5 m, clip to [0, 3.5], divide by 3.5.
   If the LiDAR is not at the robot centre, transform ranges to the centre.
2. **Goal distance.** `d = ||goal_xy - robot_xy||`, normalised by `6*sqrt(2)` and clipped to 1.
   For larger real rooms this normalisation must be revisited (it saturates beyond 8.49 m).
3. **Goal bearing.** `alpha = wrap(atan2(gy - y, gx - x) - yaw)`, provide `sin(alpha)`, `cos(alpha)`.
4. Concatenate to an 11-element `float32` vector in exactly this order.

## 3. Output

| Quantity | Topic | Message | Mapping |
|---|---|---|---|
| Velocity command | `/cmd_vel` | `geometry_msgs/msg/Twist` (or `TwistStamped` for Jazzy controllers) | `linear.x = (u_v + 1)/2 * 0.35`, `angular.z = u_w * 1.5`, all other fields 0 |

## 4. Frames

`map` -> `odom` -> `base_link` -> `laser` (REP 105). Policy angles are measured in `base_link`,
counter-clockwise positive, x forward - identical to the simulator convention.

## 5. Timing and freshness

* Control period 0.2 s (5 Hz), the same `dt` used in training. A timer drives inference; each
  tick uses the latest scan and pose.
* Freshness: discard and command zero velocity if the scan is older than 0.3 s or the pose older
  than 0.3 s (use `header.stamp`, with `use_sim_time:=true` in Gazebo).
* Gazebo stepping: for training in Gazebo, the environment wrapper must advance simulation by
  exactly 0.2 s per `step()` (e.g. pause/step service of the Gazebo world) to keep the MDP
  timing identical.

## 6. Reset behaviour (Gazebo training only)

`reset()` must: stop the robot, teleport it to the start pose (`/world/<name>/set_pose`
service), spawn/move obstacles for the scenario, set the goal, wait for one fresh scan, and
return the first observation. Outcomes: success = goal distance <= 0.25 m; collision = contact
sensor / bumper or minimum scan range <= robot radius + margin; timeout = 250 control steps.

## 7. Safety constraints (physical robot)

* Velocity limits enforced again in the node (0.35 m/s, 1.5 rad/s), plus acceleration limits.
* Emergency stop: zero velocity if any scan reading in the front sector < 0.25 m, if data are
  stale, or if the policy outputs non-finite values.
* Hardware e-stop and supervised testing only; the policy has never been validated outside the
  2D simulator.
* Sim-to-real gaps to address: sensor noise and dropout, latency, wheel slip, non-circular
  obstacles, LiDAR blind spots, robot dynamics.

## 8. Comparison baseline

Nav2 (planner + controller) is the planned classical baseline for Week 11, evaluated on the same
Gazebo worlds and the same success / collision / timeout definitions.
