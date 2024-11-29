from setuptools import find_packages, setup

setup(
    name="dqn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dqn-train=dqn.train:main",
        ],
    },
)
