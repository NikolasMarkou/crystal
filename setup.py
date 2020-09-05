# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="crystal",
    version="0.1.0",
    description="N-dimensional simplex creation",
    long_description=readme,
    author="Nikolas Markou",
    author_email="nikolasmarkou@gmail.com",
    url="https://github.com/NikolasMarkou/crystal",
    license=license,
    packages=find_packages(exclude=("tests", "docs"))
)
