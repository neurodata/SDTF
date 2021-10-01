from setuptools import setup, find_packages

REQUIREMENTS = ["numpy", "scikit-learn", "scipy"]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="spdt",
    version="0.0.2",
    author="Haoyin Xu",
    author_email="haoyinxu@gmail.com",
    description="Exploring streaming options for decision tree classifiers",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/PSSF23/SPDT/",
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(),
)
