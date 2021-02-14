import os

from setuptools import find_packages, setup

__version__ = "0.0.17"


def load_requires_from_file(fname):
    if not os.path.exists(fname):
        raise IOError(fname)
    return [pkg.strip() for pkg in open(fname, "r")]


setup(
    name="cpnet",
    version=__version__,
    author="Sadamori Kojaku",
    author_email="freesailing4046@gmail.com",
    description="Algorithm for finding multiple core-periphery pairs in networks",
    long_description="Algorithm for finding multiple core-periphery pairs in networks",
    url="https://github.com/skojaku/core-periphery-detection",
    packages=find_packages("cpnet"),
    install_requires=load_requires_from_file("requirements.txt"),
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
