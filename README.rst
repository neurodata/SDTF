
Streaming Decision Trees & Forests
==================================


.. image:: https://img.shields.io/badge/arXiv-2110.08483-red.svg?style=flat
  :target: https://arxiv.org/abs/2110.08483
  :alt: arXiv


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5557864.svg
  :target: https://doi.org/10.5281/zenodo.5557864
  :alt: DOI


.. image:: https://img.shields.io/pypi/v/sdtf.svg
  :target: https://pypi.org/project/sdtf/
  :alt: PyPI version


.. image:: https://circleci.com/gh/neurodata/SDTF/tree/main.svg?style=shield
  :target: https://circleci.com/gh/neurodata/SDTF/tree/main
  :alt: CircleCI


.. image:: https://img.shields.io/netlify/b47deb03-9e70-4684-a0a1-bbafdbcf6d49
  :target: https://app.netlify.com/sites/sdtf/deploys
  :alt: Netlify


.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg
  :target:
  :alt: Python


.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black
  :alt: Code style: black


.. image:: https://img.shields.io/badge/License-MIT-blue
  :target: https://opensource.org/licenses/MIT
  :alt: License


.. image:: https://img.shields.io/pypi/dm/sdtf.svg
  :target: https://pypi.org/project/sdtf/#files
  :alt: Downloads


.. doc-start


Exploring streaming options for decision trees and random forests.

The package includes two ensemble implementations (Stream Decision Forest and Cascade Stream Forest).

Based on ``scikit-learn`` `fork <https://github.com/neurodata/scikit-learn/tree/stream>`_.

Install
-------

You can manually download the latest version of ``SDTF`` by cloning the repository:

.. code-block::

  git clone https://github.com/neurodata/SDTF
  cd SDTF
  python setup.py install

Or install the stable version through ``pip``\ :

.. code-block::

  pip install sdtf

Package Requirements
--------------------

The ``SDTF`` package requires a ``scikit-learn`` fork for the ``partial_fit`` functionality,
which you can install manually:

.. code-block::

  git clone https://github.com/neurodata/scikit-learn -b stream --single-branch
  cd scikit-learn
  python setup.py install

The above local setup requires the following packages:


* ``cython``
* ``numpy``
* ``scipy``


.. doc-end


Relevant Repos
--------------


* `huawei-noah/streamDM <https://github.com/huawei-noah/streamDM>`_
* `soundcloud/spdt <https://github.com/soundcloud/spdt>`_
* `online-ml/river <https://github.com/online-ml/river>`_
* `scikit-garden/scikit-garden <https://github.com/scikit-garden/scikit-garden>`_

Relevant Papers
---------------


* `Very Fast Decision Tree <https://dl.acm.org/doi/10.1145/347090.347107>`_
* `Online Bagging and Boosting <https://ieeexplore.ieee.org/document/1571498>`_
* `Leveraging Bagging for Evolving Data Streams <https://link.springer.com/chapter/10.1007/978-3-642-15880-3_15>`_
* `Ensemble Learning for Data Stream Classification <https://dl.acm.org/doi/10.1145/3054925>`_
* `Adaptive Random Forests <https://link.springer.com/article/10.1007/s10994-017-5642-8>`_
* `Streaming Random Forests <https://ieeexplore.ieee.org/document/4318108>`_
* `Streaming Parallel Decision Tree <https://www.jmlr.org/papers/v11/ben-haim10a.html>`_
* `Mondrian Forests <https://papers.nips.cc/paper/5234-mondrian-forests-efficient-online-random-forests.pdf>`_
