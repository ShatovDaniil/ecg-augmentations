from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ecg-augmentations",
    version="1.0.0",
    author="Daniil",
    description="Library for ECG signal augmentation compatible with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShatovDaniil/ecg-augmentations.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "wfdb>=3.4.0",
        "h5py>=3.0.0",
        "matplotlib>=3.3.0",
    ],
)