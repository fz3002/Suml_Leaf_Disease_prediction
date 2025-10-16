from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Suml_Leaf_Disease_prediction",
    version="0.1",
    url="https://github.com/fz3002/Suml_Leaf_Disease_prediction",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
)