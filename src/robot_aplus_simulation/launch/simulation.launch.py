"""One robot, one world; Ctrl+C stops the processes owned by this launch."""
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
import os
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    share = Path(get_package_share_directory('robot_aplus_simulation'))
    world = LaunchConfiguration('world').perform(context)
    if world not in ('baseline', 'aplus_indoor_world'):
        raise ValueError('world must be baseline or aplus_indoor_world')
    description = xacro.process_file(str(share / 'urdf/simulation.urdf.xacro')).toxml()
    gui = LaunchConfiguration('gui').perform(context) == 'true'
    args = ['gz', 'sim', '-r', '-v', '3']
    if not gui:
        args += ['-s', '--headless-rendering']
    args += ['--gui-config', str(share / 'config/gui.config'),
             str(share / 'worlds' / (world + '.sdf'))]
    plugin_path = str(Path(get_package_share_directory(
        'robot_aplus_simulation')).parents[1] / 'lib')
    env = dict(os.environ)
    env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = plugin_path + \
        os.pathsep + env.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    return [
        ExecuteProcess(cmd=args, output='screen', additional_env=env),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description, 'use_sim_time': True}], output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': str(share / 'config/bridge.yaml'), 'use_sim_time': True}],
             output='screen'),
        Node(package='ros_gz_sim', executable='create',
             arguments=['-world', 'aplus', '-name', 'robot_aplus', '-topic', 'robot_description',
                        '-x', '0', '-y', '0', '-z', '0.005'], output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='aplus_indoor_world'),
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false']),
        OpaqueFunction(function=start),
    ])
