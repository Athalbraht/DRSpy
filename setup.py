from setuptools import setup

import DRSpy

deps = ["click", "pandas", "matplotlib", "scipy"]

setup(
    name=DRSpy.__name__,
    description="Data loader for DRS4-psi",
    version=DRSpy.__version__,
    url=DRSpy.__url__,
    author=DRSpy.__author__,
    license="MIT",
    requires=deps,
    install_requires=deps,
    packages=["DRSpy", "DRSpy.plugins"],
    entry_points={"console_scripts": ["drspy = DRSpy.main:main"]},
)
