# PyNav

Code used to perform the experiments in the paper "Clock-state olfactory search in turbulent flows using Q-learning: The geometry of plume recovery". Experiment scripts are provided in the [experiments](./experiments/) folder.

Instructions for setting up the environment and installing dependencies are provided below.

## :package: Installation

To use this library, you need to install it. We recommend creating a new Conda environment to manage dependencies.

In particular, this library supports Python >= 3.9 and all experiments were performed using Python 3.10. You can create and activate a Conda environment as follows:

```bash
conda create -n pynav python=3.10
conda activate pynav
```
Then, clone this repository 
```bash
git clone https://github.com/PimLb/pynav.git
cd pynav
```
Finally, install the library using the `setup.py` file
```bash
pip install .
```
The experiments were run using Python 3.10.13 and NumPy 1.26.

## :bar_chart: Dataset

The dataset used to perform the experiments can be downloaded from Zenodo at the following link: https://zenodo.org/records/20001730