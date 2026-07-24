# %% [markdown]
# # Elastic Iceberg
# 
# This tutorial shows how to get kraken running, applied to the simple case of an elastic iceberg
#
# This file can also be run in parrallel using `mpirun`
#
# First, we import kraken, as well as mesh from dolfinx, mpi4py and numpy

# %%
import kraken as kr
from mpi4py import MPI
import numpy as np
from dolfinx import mesh


# %% [markdown]
# We then define the length and height of the iceberg. We will later impose a symmetry boundary condition on the left boundary, so the length here represents half the true iceberg length
# We also define the regularisation length, and the size of the mesh which we set to be half the regularisation length. 
#%%

length = 1e3
height = 400
l = 8
cell_size = l/2

# %% [markdown]
# We now create the mesh using built in dolfinx tools.
#
# Kraken works in non-dimensional units, and non-dimensionalise by the height of the domain,
 
#%%

nx = int(length/cell_size)
nz = int(height/cell_size)
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [length/height, 1.0]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)


# %% [markdown]
# All Kraken simulations take place within this simulation class, which contains the logic for the fixed point iteration and the parameters
#%%

model = kr.base.Simulation(msh)

# %% [markdown]
# Within the model is the params class, which calculate non-dimensional numbers from physical constants.
# We update the height, regularisation length, domain length and sea-level based on the problem specifications
#%%

model.params.H.value = height
model.params.l.value = l
model.params.length.value = length
model.params.sea_level.value = model.params.H.value*0.9

# %% [markdown]
# We now define our boundary conditions. To impose a symmetry condition we first define our left boundary as a function of x
# Kraken provides a number of functions to impose different types of boundary conditions. They are specified as lambda functions to act on the function space
# Here we impose a zero dirichlet boundary condition on the x component of displacement, which is accessed through V.sub(0), acting on the left boundary
# We impose no boundary condition on the damage variable
#%%
def left_boundary(x):
    return np.isclose(x[0], 0)

u_bc = lambda V: [kr.boundaryconditions.get_zero_bc(V.sub(0), left_boundary)]
d_bc = lambda V: []


# %% [markdown]
# We now call model.setup(...) to build the finite element spaces and setup the solvers. We pass into this the
# momentum and damage model we want to use. We also pass in the momentum and damage boundary conditions as a tuple. 
#%%
model.setup(kr.momentum.elastic.Elasticity,
            kr.damage.higherorder.AT2,
            [u_bc, d_bc])


# %% [markdown]
# We now perform the fixed point iteration, iteratively solving the elastic and damage problem until the change in the damage variable reaches a sufficiently small tolerance, set by model.tol 
# This step can take a while, especially in serial
#%%
model.fixed_point()


# %% [markdown]
# Finally, we save the solution as and xdmf file, which can be opened in parrallel.
# Write xdmf can save arbitray ufl expressions as well as functions
#%%
kr.plotting.write_xdmf("./outputs/elastictest.xdmf",
                            msh, 
                            [model.momentum.u, model.damage.d],
                            ["u", "d"])
