#%%

import numpy as np
from dolfinx import mesh, io, log, default_scalar_type
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
import icebergmesh
import gmsh

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


true_length = 16e3
true_height = 300

material = Material_no_uc()
material.τ = 3600*24
# material.L = true_height    
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.l = 0.5/material.L


Hw = material.ρi/material.ρw*nondim_height
#%%
model = icebergmesh.create_iceberg_mesh(true_length, true_height, material)
msh,ct,ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

gmsh.finalize()
# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
#                             [nx,nz], mesh.CellType.quadrilateral)


#%%
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []
#%%

# dh = None
# vh,dh = monolithic.solve(msh,bc,material,H)
# for i in range(1):
#     vh = elasticity.solve(msh,symm_bc,material,dh)
#     dh,H = pf.solve(msh,no_bc,vh,material)
vh = None; dh = None
uh = None; ph = None
H = 0.0

g0 = 6.6
steps = 40
for i in range(steps):
    material.g = g0 + i*(9.81-g0)/(steps-1)
    print(material.g)
    # vh, dh = eb.fixed_point(msh, [symm_bc,no_bc], material, dh)
    vh, dh = pf.minimisation(msh, [symm_bc,no_bc], material, dh, vh, Hprev=H,tol=1e-8)
    H = pf.history_function(pf.ε(vh),material,H)
    utilities.write_xdmf("outputs/iceberg" + str(i) + ".xdmf",msh,\
                    [vh,dh],\
                    ["v","d"],t=i)
# material.ψcritstar = material.ψcritstar*5e4
# for i in range(100):
#     vh,dh = eb.fixed_point(msh, [symm_bc,no_bc], material, dh, max_its=2)
#     # vh,dh = pf.minimisation(msh, [symm_bc,no_bc], material, max_its = 1)
#     # uh,ph = stokes.solve(msh,symm_bc,vh,material,1.0,dh,u=uh,p=ph)
#     uh = vh
#     utilities.move_mesh(msh,uh,1.0)



#     utilities.write_xdmf("outputs/iceberg" + str(i) + ".xdmf",msh,\
#                     [vh,uh,dh],\
#                     ["v","u","d"],t=i)


#%%

σ_e = ufl.dev(pf.stress(vh,material.ν))
# σ_v = stokes.viscosity(uh,material.n,1e-8)*pf.ε(uh)

# ψ = free_energy(vh,material.ν)
ψ = pf.free_energy_plus(pf.ε(vh),material.ν)
ψplus = pf.free_energy_plus(pf.stress(vh,material.ν),material.ν)
ψplusp = pf.positive_part(pf.free_energy_plus(pf.ε(vh),material.ν)-material.ψcritstar)


from invariants import matrix_function
λ,E = invariants.eigenstate(pf.ε(vh)) 


# if MPI.COMM_WORLD.rank == 0:
#     utilities.plot_damage_state(vh,dh)

utilities.write_vtk("outputs/pf.pvd",msh,\
                    [vh,ψ,dh,λ[0],λ[1],σ_e,ψplus,ψplusp],\
                    ["v","ψ","d","λ1","λ2","σ_e","ψplus","ψplusp"])



# %%
