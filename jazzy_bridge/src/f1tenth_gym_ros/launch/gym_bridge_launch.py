from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, yaml

def generate_launch_description():
    default_cfg = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config', 'sim.yaml'
    )

    cfg_arg = DeclareLaunchArgument(
        'config',
        default_value=default_cfg,
        description='Path to master config (with bridge.ros__parameters inside)'
    )

    def launch_setup(context, *args, **kwargs):
        cfg_path = LaunchConfiguration('config').perform(context)
        with open(cfg_path, 'r') as f:
            master = yaml.safe_load(f)

        params = master.get('bridge', {}).get('ros__parameters', {})
        if not params:
            raise RuntimeError(f"'bridge.ros__parameters' not found in {cfg_path}")

        # Basic bridge config
        has_opp = int(params.get('num_agent', 1)) > 1
        ego_ns  = params.get('ego_namespace', 'ego_racecar')
        opp_ns  = params.get('opp_namespace', 'opp_racecar')
        map_yaml = params['map_path'] + '.yaml'

        # Controllers (defaults: ego gap-follow, opp gap-follow if present)
        ego_ctrl = str(params.get('ego_controller', 'gap_follow')).lower()
        opp_ctrl = str(params.get('opp_controller', 'gap_follow')).lower() if has_opp else 'none'

        # Bridge node (pass the dict, not a params-file path)
        bridge_node = Node(
            package='f1tenth_gym_ros',
            executable='gym_bridge',
            name='bridge',
            parameters=[params],
        )

        rviz_layout = '2_agents.rviz' if has_opp else 'gym_bridge.rviz'
        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                          'launch', rviz_layout)]
        )

        map_server_node = Node(
            package='nav2_map_server',
            executable='map_server',
            parameters=[
                {'yaml_filename': map_yaml},
                {'topic': 'map'},
                {'frame_id': 'map'},
                {'output': 'screen'},
                {'use_sim_time': True},
            ],
        )

        nav_lifecycle_node = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['map_server']}]
        )

        ego_robot_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='ego_robot_state_publisher',
            parameters=[{'robot_description': Command(['xacro ',
                os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                             'launch', 'ego_racecar.xacro')])}],
            remappings=[('/robot_description', 'ego_robot_description')]
        )

        nodes = [rviz_node, bridge_node, nav_lifecycle_node, map_server_node, ego_robot_publisher]

        # --- Ego controller: GAP-FOLLOW (for testing) or RL (later)
        if ego_ctrl == 'gap_follow':
            nodes.append(Node(
                package='gap_follow',
                executable='reactive_node',
                namespace=ego_ns,
                name='gap_follow_ego',
                parameters=[{'use_sim_time': True}],
            ))
        elif ego_ctrl == 'rl':
            nodes.append(Node(
                package='rl_car_controller',
                executable='rl_agent_node',
                namespace=ego_ns,
                name='rl_car_controller',
                parameters=[{'use_sim_time': True}],
                # If your RL node needs a path, pass it:
                # arguments: ['--config', cfg_path],
            ))

        # --- Opponent controller (only if num_agent > 1)
        if has_opp:
            opp_robot_publisher = Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='opp_robot_state_publisher',
                parameters=[{'robot_description': Command(['xacro ',
                    os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                 'launch', 'opp_racecar.xacro')])}],
                remappings=[('/robot_description', 'opp_robot_description')]
            )
            nodes.append(opp_robot_publisher)

            if opp_ctrl == 'gap_follow':
                nodes.append(Node(
                    package='opp_gap',
                    executable='opp_reactive_node',
                    namespace=opp_ns,
                    name='opp_gap',
                    parameters=[{'use_sim_time': True}],
                ))
            elif opp_ctrl == 'rl':
                nodes.append(Node(
                    package='rl_car_controller',
                    executable='rl_agent_node',
                    namespace=opp_ns,
                    name='rl_opp_controller',
                    parameters=[{'use_sim_time': True}],
                ))
            # 'none' -> no controller; beware: bridge needs both actions to step

        return nodes

    return LaunchDescription([cfg_arg, OpaqueFunction(function=launch_setup)])
