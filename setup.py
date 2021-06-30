from setuptools import setup

setup(
    name="pd4ml",
    version="0.3",
    description="setup for downloading data comparison data",
    url="https://github.com/erikbuh/erum_data_data",
    author="Erik Buhmann",
    author_email="erik.buhmann@uni-hamburg.de",
    license="MIT",
    packages=["pd4ml"],
    install_requires=[
        "numpy >= 1.14.0",
        "six >= 1.10.0",
        "rich >= 9.10.0",
        "requests >= 2.24.0",
        "pandas >= 1.2.2",
        "uproot-methods >= 0.9.2",
    ],
    python_requires="~=3.8",
    zip_safe=False,
)
