from setuptools import setup, find_packages

setup(
    name="lidar_processor", 
    version="0.1.0",
    packages=find_packages(),
    package_data={"lidar_processor": ["cluster_kd_tree.pkl"]},
    install_requires=[
        "numpy",
        "cupy-cuda12x",  # Adjust based on CUDA version
        "scipy",
        "matplotlib",
        "opencv-python",
        "open3d",
    ],
)
