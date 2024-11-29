#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
import elasticity
from material import MaterialProperties, Material_no_uc
import invariants
from boundaryconditions import get_zero_bc
import stokes
import poisson
import phasefield
import utilities
from common import *

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


true_length = 16e3
true_height = 300

material = Material_no_uc(g=0.1)

nondim_length = true_length/material.L
nondim_height = true_height/material.L


Hw = material.ρi/material.ρw*nondim_height

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
                            [100,10], mesh.CellType.triangle)


material.set_l_from_mesh(msh)
# 
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []

bc = symm_bc


vh = elasticity.solve(msh,bc,material)
uh, ph = stokes.solve(msh,bc,vh,material,1.0)
dh,H = phasefield.solve(msh,vh,material,0.0)



# vh = elasticity.solve(msh,material,bc,dh)

σ_e = ufl.dev(elasticity.stress(vh,material.ν))
σ_v = viscosity(uh,material.n,1e-8)*ε(uh)

ψ = phasefield.free_energy_plus(vh,material.ν)


from invariants import matrix_function
λ,E = invariants.eigenstate(matrix_function((ε(vh)),phasefield.positive_part))


utilities.write_vtk("outputs/phasefield.pvd",msh,\
                    [vh,uh,ψ,dh,λ[0],λ[1],σ_e,σ_v,ε(uh)],\
                    ["v","u","ψ","d","λ1","λ2","σ_e","σ_v","ε(u)"])


# %%
