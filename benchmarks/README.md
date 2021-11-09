## Example Usage

Run all classifier benchmarks for CIFAR-10:

```
cd code
python cifar10.py -all
```

or run specified classifier benchmarks for CIFAR-10

```
cd code
python cifar10.py -sdf -csf
```

The `pendigits` [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/) needs to be downloaded to this folder manually.

The `splice` [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/) requires the processed file: `splice.csv`. Please refer to the `arXiv` [preprint](https://arxiv.org/abs/2110.08483) for processing details.
