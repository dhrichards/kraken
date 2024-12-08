#%%

import numpy as np
from dolfinx import mesh, fem, log, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
import elasticity
from material import MaterialProperties, Material_no_uc
import invariants
from boundaryconditions import get_zero_bc
import stokes
import phasefield as pf
import utilities
import energybased as eb

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


true_length = 16e3
true_height = 300

material = Material_no_uc()
material.τ = 365*24*3600/12
# material.L = true_height    
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
#     dh,H = pf.solve(msh,no_bc,vh,material)

# material.ψcritstar = material.ψcritstar*5e4

vh,dh = eb.fixed_point(msh, [symm_bc,no_bc], material)
# vh,dh = pf.minimisation(msh, [symm_bc,no_bc], material, max_its = 1)
uh,ph = stokes.solve(msh,symm_bc,vh,material,1.0,dh)
uh_old, ph_old = stokes.solve(msh,symm_bc,vh,material,1.0)
#%%
# dt = 1.0
# log.set_log_level(log.LogLevel.INFO)
# # uh, ph = stokes.solve(msh,symm_bc,vh,material,dt)

# uh = None
# for i in range(5):
#     vh,dh = eb.fixed_point(msh, [clamped_bc,no_bc], material)
#     uh,ph = stokes.solve(msh,clamped_bc,vh,material,dt,u=uh)
#     # utilities.move_mesh(msh,uh,dt)


#%%

σ_e = ufl.dev(pf.stress(vh,material.ν))
σ_v = stokes.viscosity(uh,material.n,1e-8)*pf.ε(uh)

# ψ = free_energy(vh,material.ν)
ψ = pf.free_energy(vh,material.ν)
ψplus = pf.free_energy_plus(vh,material.ν)
ψplusp = pf.positive_part(pf.free_energy_plus(vh,material.ν)-material.ψcritstar)


from invariants import matrix_function
λ,E = invariants.eigenstate(pf.ε(vh)) 


# if MPI.COMM_WORLD.rank == 0:
#     utilities.plot_damage_state(vh,dh)

utilities.write_vtk("outputs/pf.pvd",msh,\
                    [vh,uh,ψ,dh,λ[0],λ[1],σ_e,σ_v,ψplus,ψplusp,uh_old],\
                    ["v","uh","ψ","d","λ1","λ2","σ_e","σ_v","ψplus","ψplusp","uh_old"])



# %%
