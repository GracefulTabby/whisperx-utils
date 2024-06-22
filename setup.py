import os
from setuptools import find_packages, setup

setup(
    name="whisperx-utils",
    py_modules=["whisperx_utils"],
    version="0.1.0",
    description="Utility functions for WhisperX transcription",
    readme="README.md",
    python_requires=">=3.8",
    author="GracefulTabby",
    url="https://github.com/GracefulTabby/whisperx-utils",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "whisperx",
        "tqdm",
        "pandas",
        "numpy",
        "python-dotenv",
    ],
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
