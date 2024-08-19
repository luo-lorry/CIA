# CIA
This repo contains code implementation for Conformalized Interval Arithmetic (CIA), which extends Conformal Prediction's coverage guarantees to the arithmetic operations of variables, such as summation.

## Introduction

The main repository contains the code for experiments on subsets without overlap, specifically for *bike*, *community*, and *meps* datasets.

## Code Structure

The code is organized into two main files:

* `main.py`: Contains the code for running experiments on the bike, community, and meps datasets.
* `plot_results.py`: Used to plot the results presented in the main text and appendix of the paper.

## Dependencies

The code requires the following dependencies:

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Running the Code

To run the code, simply execute the `main.py` file using Python. This will run the experiments on the specified datasets.

## Split Conformal Method

The Split Conformal Method uses a neural network as the base conditional mean predictor. The network consists of three fully connected layers with a hidden dimension of 64. This setting is based on the work of [1].

## Conformal Quantile Regression

Conformal Quantile Regression uses the quantile regression forests [2] as the quantile predictor.

## Plotting Results

The `plot_results.py` file is used to plot the results presented in the main text and appendix of the paper. This file takes the output of the `main.py` file as input and generates the plots.

## Experiments with Overlap

The folder `overlap` contains code and data for experiments for subsets with overlaps, specifically the road traffic in Anaheim and Chicago. Run `main_overlap.py` to run the experiments.

### Directed Graph Autoencoder

The files `digae_traffic.py` and `autoencoder.py` contain code that implements a directed graph autoencoder that predicts edge weights based on node (and edge) features.

### Required Packages

Some required packages include:

* `networkx`
* `geopandas`
* `pandas`
* `numpy`
* `matplotlib`
* `torch`
* `torch_geometric`

### Implementation

* `DirectedGNN`, `train_digae`, `test_digae`, `train_digae_quantile_regression`, and `test_digae_quantile_regression` are implemented in `digae_traffic.py`
* `GAE` and `DirectedEdgeDecoder` are implemented in `autoencoder.py`

## Data

### Meps Data

Due to copyright/usage rules, The Medical Expenditure Panel Survey (MPES) data can be downloaded following the instructions in https://github.com/yromano/cqr/blob/master/get_meps_data/README.md.

Please put the downloaded and processed .csv files in the `\common\datasets` folder.

## Acknowledgments

We thank the following GitHub repositories for their code:

* [CQR](https://github.com/yromano/cqr): For the implementation of the Split Conformal Method
* [QRF](https://github.com/zillow/quantile-forest): For the implementation of Quantile Regression Forests
* [GNN](https://github.com/000Justin000/gnn-residual-correlation): For the processing of road traffic datasets
* [CF-GNN](https://github.com/snap-stanford/conformalized-gnn): For the implementation of conformalized graph neural network

## References
[1] [**Conformal Quantile Regression**](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) 

[2] [**Quantile Regression Forests**](http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf)
