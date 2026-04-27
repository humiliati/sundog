from setuptools import setup, find_packages

setup(
    name='sundog-leisure',
    version='0.1.0',
    description='Embodied resonance through tinkering - Sundog Alignment Theorem extension',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
    ],
    extras_require={
        'mujoco': ['mujoco>=2.3.0'],
        'viz': ['matplotlib>=3.4.0'],
        'dev': ['pytest>=6.0.0'],
    },
    entry_points={
        'console_scripts': [
            'train-leisure=scripts.train_leisure:main',
        ],
    },
)










