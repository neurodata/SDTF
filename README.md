# Streaming Decision Trees & Forests

[![arXiv](https://img.shields.io/badge/arXiv-2110.08483-red.svg?style=flat)](https://arxiv.org/abs/2110.08483)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5557864.svg)](https://doi.org/10.5281/zenodo.5557864)
[![PyPI version](https://img.shields.io/pypi/v/sdtf.svg)](https://pypi.org/project/sdtf/)
[![CircleCI](https://circleci.com/gh/neurodata/SDTF/tree/main.svg?style=shield)](https://circleci.com/gh/neurodata/SDTF/tree/main)
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/sdtf.svg)](https://pypi.org/project/sdtf/#files)

Exploring streaming options for decision trees and random forests.

The package includes two ensemble implementations (Stream Decision Forest and Cascade Stream Forest).

Based on `scikit-learn` [fork](https://github.com/neurodata/scikit-learn/tree/stream).

## Install

You can manually download the latest version of `SDTF` by cloning the repository:

```
git clone https://github.com/neurodata/SDTF
cd SDTF
python setup.py install
```

Or install the stable version through `pip`:

```
pip install sdtf
```

## Package Requirements

The `SDTF` package requires a `scikit-learn` fork for the `partial_fit` functionality,
which you can install manually:

```
git clone https://github.com/neurodata/scikit-learn -b stream --single-branch
cd scikit-learn
python setup.py install
```

The above local setup requires the following packages:

- `cython`
- `numpy`
- `scipy`

## Relevant Repos

- [huawei-noah/streamDM](https://github.com/huawei-noah/streamDM)
- [soundcloud/spdt](https://github.com/soundcloud/spdt)
- [online-ml/river](https://github.com/online-ml/river)
- [scikit-garden/scikit-garden](https://github.com/scikit-garden/scikit-garden)

## Relevant Papers

- [Very Fast Decision Tree](https://dl.acm.org/doi/10.1145/347090.347107)
- [Online Bagging and Boosting](https://ieeexplore.ieee.org/document/1571498)
- [Leveraging Bagging for Evolving Data Streams](https://link.springer.com/chapter/10.1007/978-3-642-15880-3_15)
- [Ensemble Learning for Data Stream Classification](https://dl.acm.org/doi/10.1145/3054925)
- [Adaptive Random Forests](https://link.springer.com/article/10.1007/s10994-017-5642-8)
- [Streaming Random Forests](https://ieeexplore.ieee.org/document/4318108)
- [Streaming Parallel Decision Tree](https://www.jmlr.org/papers/v11/ben-haim10a.html)
- [Mondrian Forests](https://papers.nips.cc/paper/5234-mondrian-forests-efficient-online-random-forests.pdf)
