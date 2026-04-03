from setuptools import setup

setup(
    name="polygon-shim",
    version="0.1",
    packages=["Polygon", "lanms"],
    install_requires=["shapely"],
)
