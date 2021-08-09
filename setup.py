#!/usr/bin/env python

with open('README.md', 'r') as f:
    long_description = f.read()

version = {}
with open("graphsite/version.py", "r") as stream:
    exec(stream.read(), version)
    
from setuptools import setup
setup(
    name="graphsite",
    version=version["__version__"],
    author="Wentao Shi",
    author_email="wentao771@gmail.com",
    description="Compute graph representations of protein binding sites",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shiwentao00/Graphsite",    
    install_requires=['numpy', 'pandas', 'biopandas', 'scipy'],
    packages=['graphsite'],
    python_requires=">=3.7",
)
