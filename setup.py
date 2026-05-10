from setuptools import setup, find_packages

setup(
    name="minicv",
    version="0.1.0",
    description="A minimal image processing library built from scratch using NumPy and Matplotlib.",
    author="Abdelghaffar",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
    ],
)
