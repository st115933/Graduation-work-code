from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _declare_args(arg_defs):
    """
    Helper to declare launch arguments.
    arg_defs: list of tuples (name, default, description)
    """
    return [DeclareLaunchArgument(name, default_value=default, description=desc)
            for name, default, desc in arg_defs]


def _make_node(pkg, exe, node_name, namespace, params, remappings):
    """
    Helper to create a Node action.
    params: dict of parameter names to LaunchConfiguration
    remappings: list of (from_topic, to_topic)
    """
    return Node(
        package=pkg,
        executable=exe,
        name=node_name,
        namespace=namespace,
        parameters=[params],
        remappings=remappings
    )


def generate_launch_description():
    # argument definitions: (name, default, description)
    arg_definitions = [
        ('yolomodel', 'best.pt', 'yolomodel path or name'),
        ('tracker', 'bytetrack.yaml', 'Tracker config file'),
        ('device', 'cpu', 'Compute device'),
        ('yoloenable', 'True', 'yoloenable detector at startup'),
        ('confidence_threshold', '0.5', 'Detection confidence confidence_threshold'),
        ('input_image_topic', '/rgb', 'Input image topic name'),
        ('namespace', 'yolo', 'Namespace for all nodes'),
    ]

    # Declare all launch args
    ld = LaunchDescription(_declare_args(arg_definitions))

    # Create LaunchConfiguration mappings
    cfg = {name: LaunchConfiguration(name) for name, _, _ in arg_definitions}

    # Define nodes to launch
    nodes_to_launch = [
        {
            'pkg': 'yolov8_ros',
            'exe': 'yolov8_node',
            'name': 'yolov8_node',
            'params': {
                'yolomodel': cfg['yolomodel'],
                'device': cfg['device'],
                'yoloenable': cfg['yoloenable'],
                'confidence_threshold': cfg['confidence_threshold'],
            },
            'remaps': [('image_raw', cfg['input_image_topic'])]
        },
        {
            'pkg': 'yolov8_ros',
            'exe': 'tracking_node',
            'name': 'tracking_node',
            'params': {'tracker': cfg['tracker']},
            'remaps': [('image_raw', cfg['input_image_topic'])]
        },
        {
            'pkg': 'yolov8_ros',
            'exe': 'visualization_Node',
            'name': 'visualization_Node',
            'params': {},
            'remaps': [
                ('image_raw', cfg['input_image_topic']),
                ('detections', 'tracking')
            ]
        },
    ]

    # Add Node actions to launch description
    for node_def in nodes_to_launch:
        ld.add_action(_make_node(
            pkg=node_def['pkg'],
            exe=node_def['exe'],
            node_name=node_def['name'],
            namespace=cfg['namespace'],
            params=node_def['params'],
            remappings=node_def['remaps']
        ))

    return ld

