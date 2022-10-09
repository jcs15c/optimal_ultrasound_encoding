# Optimal Ultrasound Encoding

This package contains methods to find optimal encoding sequences for use in synthetic transmit aperture ultrasound imaging. 
The module `encoding_optimization.py` contains the primary function to perform the optimization. Example parameters are provided in the `examples` folder, along with a Jupyter notebook for sample execution. Sample RF data can be found [here](https://drive.google.com/drive/folders/1fIrST8dmWBjikxwa5i3ZhP9SyMiAkwSI?usp=sharing) for download.

# Dependencies

This code requires the following dependencies, with the most recently tested version in parentheses.

- Python 3 (3.9.5)
- PyTorch (1.11.0)
- scipy (1.7.3)
- numpy (1.22.3)
- matplotlib (3.5.1)
- pandas (1.4.2)

Additionally, PyTorch can be installed with CUDA Toolkit (11.3) to run the code on GPU and dramatically increase runtime efficiency.
