from setuptools import setup, find_packages

setup(
    name="lidar_processor", 
    version="0.1.0",
    packages=find_packages(include =["lidar_processing","lidar_processing.*"]),
    package_data={"lidar_processing": ["*.pkl"]},
    include_package_data = True,
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],
)
