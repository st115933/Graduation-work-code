from pathlib import Path
from setuptools import setup, find_packages
# Define module name (should match package name)
MODULE_NAME = 'detect_bac'
ROOT_DIR = Path(__file__).resolve().parent

def prepare_assets():
    """
    准备安装所需的数据文件：
      - package.xml
      - launch 文件（如果存在）
    所有路径必须是相对路径。
    """
    # Register package with ROS ament resource index
    resource_target = ('share/ament_index/resource_index/packages', ['resource/' + MODULE_NAME])
    # Install package.xml to share directory
    pkg_xml_data = (f'share/{MODULE_NAME}', ['package.xml'])

    # Find and install launch files
    launch_source = ROOT_DIR / 'launch'
    launch_files = [str(p.relative_to(ROOT_DIR)) for p in launch_source.glob('*launch.[pxy][yma]*')]
    launch_data = [(f'share/{MODULE_NAME}/launch', launch_files)] if launch_files else []

    # Combine all asset targets
    return [resource_target, pkg_xml_data, *launch_data]


setup(
    name=MODULE_NAME,
    version="0.0.0",
    packages=find_packages(include=[MODULE_NAME]),
    data_files=prepare_assets(),
    entry_points={
        "console_scripts": [
            "area_find = detect_bac.area_find:main", 
        ]
    },
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="123",
    maintainer_email="123@g123.ru",
    description="description",      
    license="License declaration",   
    tests_require=["pytest"]
)
