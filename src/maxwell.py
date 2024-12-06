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
import energybased as eb
from common import *

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


true_length = 16e3
true_height = 300

material = Material_no_uc()
material.L = true_height    
nondim_length = true_length/material.L
nondim_height = true_height/material.L

nz = 20
cell_size = nondim_height/nz
nx = int(nondim_length/cell_size/2)


Hw = material.ρi/material.ρw*nondim_height

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
                            [nx,nz], mesh.CellType.triangle)


material.set_l_from_mesh(msh)
# 
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []


# dh = None
# vh,dh = monolithic.solve(msh,bc,material,H)
# for i in range(1):
#     vh = elasticity.solve(msh,symm_bc,material,dh)
#     dh,H = phasefield.solve(msh,no_bc,vh,material)



vh,dh = eb.fixed_point(msh, [clamped_bc,no_bc], material)

#%%
dt = material.yrs2nondimt(1/365)
uh = stokes.solve(msh,clamped_bc,vh,material,dt)


# for i in range(5):
#     vh,dh = eb.fixed_point(msh, [no_bc,no_bc], material)
#     utilities.plot_damage_state(vh,dh)
#     uh = stokes.solve(msh,no_bc,vh,material,dt)

    
#     utilities.plot_damage_state(uh,dh)





#%%

σ_e = elasticity.stress(vh,material.ν)
# σ_v = viscosity(uh,material.n,1e-8)*ε(uh)

# ψ = free_energy(vh,material.ν)
ψ = free_energy(vh,material.ν)
ψplus = phasefield.free_energy_plus(vh,material.ν)
ψplusp = phasefield.positive_part(phasefield.free_energy_plus(vh,material.ν)-material.ψcritstar)


from invariants import matrix_function
λ,E = invariants.eigenstate(ε(vh)) 


# if MPI.COMM_WORLD.rank == 0:
#     utilities.plot_damage_state(vh,dh)


utilities.write_vtk("outputs/phasefield.pvd",msh,\
                    [vh,ψ,dh,λ[0],λ[1],σ_e,ψplus,ψplusp],\
                    ["v","ψ","d","λ1","λ2","σ_e","ψplus","ψplusp"])



# %%
