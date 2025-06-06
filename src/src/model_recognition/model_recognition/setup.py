from setuptools import setup, find_packages

# Define the project name and version number
PROJECT_NAME = 'model_recognition'
VERSION = '0.0.0'

# Configure dictionary
configurations = {
    "name": PROJECT_NAME,
    "version": VERSION,
    "packages": find_packages(include=[PROJECT_NAME]),
    "data_files": [
        # Register the package to the ament index
        ('share/ament_index/resource_index/packages', ['resource/' + PROJECT_NAME]),
        # Copy package.xml to the installation directory
        (f'share/{PROJECT_NAME}', ['package.xml']),
    ],
    "entry_points": {
        'console_scripts': [
            'yolov8_node = model_recognition.yolov8_node:main',
            'visualization_Node = model_recognition.VisualizationNode:main',  
            'tracking_node = model_recognition.tracking_node:main'
        ]
    },
    "install_requires": ['setuptools'],
    "zip_safe": True,
    "maintainer": '123',
    "maintainer_email": '123@123.ru',
    "description": 'ROS2',
    "license": 'License declaration',  
    "tests_require": ['pytest']
}

#  Execute function
setup(**configurations)
