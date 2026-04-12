from setuptools import setup, find_packages

setup(
    name='ml-interview-prep',
    version='1.0.0',
    description='Clean NumPy + PyTorch implementations of core ML algorithms for interview preparation.',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21',
        'torch>=1.10',
    ],
)
