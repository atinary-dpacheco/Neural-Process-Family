# !/usr/bin/env python


from setuptools import find_packages, setup

__version__ = "0.1.0"


setup(
    name="neural_process_family",
    version=__version__,
    description="Implementation of Yann Dubois about NP families",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "skorch==0.8 ",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "matplotlib",
        "tqdm",
        "scikit-learn==1.2.2",
        "joblib==1.3.2",
        "h5py==3.10.0",
        "seaborn",
        "scikit-image",
        "imageio<=2.27",
    ],
    include_package_data=True,
)
