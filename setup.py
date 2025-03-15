from setuptools import setup, find_packages

setup(
    name="trainselpy",
    version="0.1.0",
    author="Python Implementation of TrainSel",
    author_email="",
    description="Training Population Selection Optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.2.0",
        "joblib>=0.16.0",
    ],
    include_package_data=True,
)