import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# Find mgc version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "sdtf", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]

REQUIREMENTS = ["numpy", "scikit-learn", "scipy"]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")
        version = "v{}".format(VERSION)

        if tag != version:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, version
            )
            sys.exit(info)


setup(
    name="sdtf",
    version=VERSION,
    author="Haoyin Xu",
    author_email="haoyinxu@gmail.com",
    description="Exploring streaming options for decision trees and random forests. Based on scikit-learn fork.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/SDTF",
    download_url="https://codeload.github.com/neurodata/SDTF/tar.gz/v" + VERSION,
    keywords=["Streaming Trees", "Machine Learning", "Decision Trees"],
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
