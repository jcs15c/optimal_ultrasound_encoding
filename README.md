[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10689039.svg)](https://doi.org/10.5281/zenodo.10689039)

# Optimal Ultrasound Encoding

This package contains methods to find optimal encoding sequences for use in synthetic transmit aperture ultrasound imaging.

This code is based on the procedures introduced in ``[Optimization of array encoding for ultrasound imaging](https://iopscience.iop.org/article/10.1088/1361-6560/ad5249/meta)'' ([arXiv](https://arxiv.org/abs/2403.00289)). 

The module `encoding_optimization.py` contains the primary function to perform the optimization and generate an optimized encoding sequence. An example execution script is provided in the `examples` folder. 

Collections of sample RF data to used for training can be found in the repository on Zenodo, along with simulated and experimentally acquired data used for validation of the model. Examples of optimized sequences can be found in the GitHub repository.

# Dependencies

This code requires the following dependencies, with the most recently tested version in parentheses.

- Python 3 (3.9.5)
- PyTorch (1.11.0)
- scipy (1.7.3)
- numpy (1.22.3)
- matplotlib (3.5.1)
- pandas (1.4.2)

Additionally, PyTorch can be installed with CUDA Toolkit (11.3) to run the code on GPU and dramatically increase runtime efficiency.
