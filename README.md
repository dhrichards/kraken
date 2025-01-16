# kraken3
Phase field visco-elastic solver in Fenicsx


## Installation

Follow the [instructions to install fenicsx in conda](https://fenicsproject.org/download/):

```commandline
conda create -n fenicsx-env python=3.12
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
conda install cuda-cudart cuda-version=12           # If using an appropriate GPU
```


