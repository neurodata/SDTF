.. SDTF documentation master file, created by
   sphinx-quickstart on Fri Oct 22 10:16:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========


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
  :target: https://www.python.org/downloads/
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


**S**\ treaming **D**\ ecision **T**\ rees & **F**\ orests: exploring streaming options for decision trees and random forests.

The package includes two ensemble implementations (**Stream Decision Forest** and **Cascade Stream Forest**).

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


Benchmarks
==========

.. toctree::
  :maxdepth: 1

  visual

API
===

.. toctree::
  :maxdepth: 1

  api
