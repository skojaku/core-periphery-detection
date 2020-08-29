from setuptools import setup, find_packages
import os

__version__ = "0.0.4"


setup(
    name="cpnet",
    version=__version__,
    author="Sadamori Kojaku",
    author_email="freesailing4046@gmail.com",
    description="Algorithm for finding multiple core-periphery pairs in networks",
    long_description="Algorithm for finding multiple core-periphery pairs in networks",
    url="https://github.com/skojaku/core-periphery-detection",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "networkx>=2.0",
        "numpy>=1.16.0",
        "simanneal>=0.4.2",
        "scipy>=1.5.2",
        "numba==0.50.0",
        "joblib>=0.16.0",
        "tqdm",
    ],
    zip_safe=False,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="network core-periphery structure",
)
