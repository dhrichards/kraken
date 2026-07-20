# kraken
Phase field visco-elastic solver in Fenicsx


## Installation

Follow the [instructions to install fenicsx](https://fenicsproject.org/download/):

Kraken currently works with dolfinx 0.9.0, please install this version of dolfinx and its dependencies. The easiest way to do this is to use conda:


```commandline
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx=0.9.0
```

Once this conda environment is set up, you can then install this from source:

```
pip install .
```
