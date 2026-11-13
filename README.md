# Autonomous Indoor Mobile Robot Navigation Using Deep Reinforcement Learning

This capstone project develops an indoor differential-drive robot for robot modeling, wheel odometry, simulation, navigation, and deep reinforcement learning (DRL). The prototype is inspired by the TurtleBot3 Waffle Pi, but it is not an exact replica.

## Project status

| Area | Current state |
| --- | --- |
| Robot description | URDF/Xacro model, collision and inertial geometry, TF frames, and RViz 2 display are available in `robot_aplus_description`. |
| Differential-drive kinematics | Coordinate and wheel sign conventions are defined; no control node is implemented. |
| Wheel odometry | Not implemented. |
| Gazebo Harmonic simulation | Not integrated or physics-tested. |
| Navigation and DRL | Not implemented. |

The repository currently contains one ROS 2 package under `src/robot_aplus_description/`. Its Xacro model is in `urdf/`, the display launch file is in `launch/`, and the RViz configuration is in `rviz/`. The root `Makefile` contains a C++ formatting check for future source code.

## Robot reference

The robot uses REP-103 body axes: `+X` forward, `+Y` left, and `+Z` up. `base_link` is the midpoint of the two drive-wheel axles. `base_footprint` is its nominal projection onto the floor. Both wheel joints rotate about `+Y`; positive wheel speed corresponds to forward rolling under the no-slip assumption.

| Prototype parameter | Value |
| --- | ---: |
| Wheel radius | `0.065 m` |
| Wheel separation (center to center) | `0.290 m` |
| Lower deck (length × width × thickness) | `0.340 × 0.240 × 0.010 m` |
| Upper deck (length × width × thickness) | `0.300 × 0.220 × 0.010 m` |
| Total modeled mass | `1.70 kg` |

The model also includes four support pillars, a rear passive ball support, and an upper LiDAR housing. All dimensions and masses are prototype assumptions, not measured hardware specifications. The Xacro properties are the source of truth for the complete parameter set.

For wheel radius `r`, wheel separation `L`, and left/right wheel angular speeds `omega_L` and `omega_R`, the intended differential-drive model is:

```text
v     = r * (omega_L + omega_R) / 2
Omega = r * (omega_R - omega_L) / L
```

Wheel odometry, when implemented, will estimate pose from measured wheel increments. It must not be treated as ground truth.

## Build and display the current robot description

Use Ubuntu 24.04 with ROS 2 Jazzy, `colcon`, `xacro`, `robot_state_publisher`, `joint_state_publisher_gui`, and RViz 2:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/robot_aplus_ws
colcon build --packages-select robot_aplus_description
source install/setup.bash
ros2 launch robot_aplus_description display.launch.py
```

The launch file starts `robot_state_publisher`, `joint_state_publisher_gui`, and RViz 2. To run without a graphical interface, use:

```bash
ros2 launch robot_aplus_description display.launch.py use_gui:=false use_rviz:=false
```

The non-GUI option runs `joint_state_publisher` with default joint positions. Joint state and TF updates describe the robot pose; they do not move a physical robot.

## Verify the description

```bash
source /opt/ros/jazzy/setup.bash
cd ~/robot_aplus_ws
xacro src/robot_aplus_description/urdf/mobile_robot.urdf.xacro > /tmp/robot_aplus.urdf
check_urdf /tmp/robot_aplus.urdf
```

After launching the display, run `ros2 run tf2_ros tf2_echo base_link lidar_link` in another sourced terminal. The expected translation is `(0, 0, 0.125) m` with zero rotation. At zero wheel angle, `base_link` to the left and right wheel frames has Y translation `+0.145 m` and `-0.145 m`, respectively. In RViz, check that the two decks, four pillars, two wheels, LiDAR housing, and rear support are visible.

## Current limitations and troubleshooting

- If ROS cannot find `robot_aplus_description`, build the package and run `source install/setup.bash` in each terminal.
- If RViz reports a missing transform, check that its Fixed Frame is `base_footprint` and that the launch process is still running. Wheel TF requires `/joint_states`.
- If a display dependency is missing, install the corresponding ROS Jazzy package, such as `ros-jazzy-xacro`, `ros-jazzy-joint-state-publisher-gui`, or `ros-jazzy-rviz2`.
- The LiDAR is a housing and frame only; it does not publish scans. The rear ball is fixed geometry, not a validated free-rolling caster.
- No Gazebo controller, simulated locomotion, odometry publisher, navigation stack, or DRL agent is provided yet. Ground contact, friction, stability, and controller behavior require separate Gazebo tests.
