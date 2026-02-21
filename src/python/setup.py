from setuptools import setup, find_packages

setup(
    name='ML-in-C',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    extras_require={
        'torch': ['torch'],
    },
)
