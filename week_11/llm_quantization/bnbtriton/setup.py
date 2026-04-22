import codecs
from setuptools import setup, find_packages

setup(
    name="bnbtriton",
    version="0.1",
    description="Triton implementation of INT8 quantization of weights and activations",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/ZhMax/bnbtriton",
    author="ZhMax",
    python_requires=">=3.10",
)