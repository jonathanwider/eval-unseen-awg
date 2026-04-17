# Evaluation code for unseen-awg

This repository accompanies the `unseen-awg` model code. It contains a set of evaluations we performed using different versions of weather generators.

To use this repository, make use of the conda environment defined in the main `unseen-awg` repository. Use pip after navigating to the base directory of eval-unseen-awg to also install the analyses:

`pip install -e .`

For practical reasons we also use `configs/paths.yaml` and `configs/variables.yaml` files here in adition to those defined in `unseen-awg`. To avoid errors, their contents should be identical to that of the configs in the `unseen-awg`.repository.