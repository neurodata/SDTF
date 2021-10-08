from setuptools import setup, find_packages

REQUIREMENTS = ["numpy", "scikit-learn", "scipy"]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="sdtf",
    version="0.0.3",
    author="Haoyin Xu",
    author_email="haoyinxu@gmail.com",
    description="Exploring streaming options for decision trees and random forests. Based on scikit-learn fork.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/SDTF",
    download_url="https://codeload.github.com/neurodata/SDTF/tar.gz/v0.0.3",
    keywords=["Streaming Trees", "Machine Learning", "Decision Trees"],
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(),
)
