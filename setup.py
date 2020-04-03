import setuptools

setuptools.setup(
    name="torchdiffeq",
    version="0.0.1",
    author="Ricky Tian Qi Chen",
    author_email="rtqichen@cs.toronto.edu",
    description="ODE solvers and adjoint sensitivity analysis in PyTorch.",
    url="https://github.com/rtqichen/torchdiffeq",
    packages=setuptools.find_packages(),
    install_requires=['torch>=0.4.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
