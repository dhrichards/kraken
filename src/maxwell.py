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
import elasticityclass as ec

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


true_length = 4e3
true_height = 300

material = Material_no_uc()
material.L = true_height
material.τ = 3600*24
# material.L = true_height    
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.l = 1.0/material.L


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

model = ec.viscoelastic_damage(msh, [symm_bc,symm_bc,no_bc], material, 1.0)

# g0 = 6.7
g0=6.85
model.material.g = g0
steps = 10
for i in range(2):
    model.material.g = g0 + i*(9.81-g0)/(steps-1)
    print(model.material.g)
   
    model.elastic_damage_fixed_point(tol=1e-6)
    utilities.write_xdmf("outputs/iceberginitial" + str(i) + ".xdmf",msh,\
                    [model.v,model.d],\
                    ["v","d"],t=i)
    

model.material.g += (9.81-g0)/(steps-1)
for i in range(2):
    
    
    model.elastic_damage_fixed_point(tol=1e-6)
    model.solve_stokes()
    

    utilities.write_xdmf("outputs/iceberg" + str(i) + ".xdmf",msh,\
                    [model.v,model.d,model.u],\
                    ["v","u","d"],t=i)

    model.update_mesh()

#%%
# vh = model.v
# σ_e = ufl.dev(pf.stress(vh,material.ν))
# # σ_v = stokes.viscosity(uh,material.n,1e-8)*pf.ε(uh)

# # ψ = free_energy(vh,material.ν)
# ψ = pf.free_energy_plus(pf.ε(vh),material.ν)
# ψplus = pf.free_energy_plus(pf.stress(vh,material.ν),material.ν)
# ψplusp = pf.positive_part(pf.free_energy_plus(pf.ε(vh),material.ν)-material.ψcritstar)


# from invariants import matrix_function
# λ,E = invariants.eigenstate(pf.ε(vh)) 


# # if MPI.COMM_WORLD.rank == 0:
# #     utilities.plot_damage_state(vh,dh)

utilities.write_vtk("outputs/pf.pvd",msh,\
                    [model.u,model.v,model.d],\
                    ["u","v","d"],t=0)



# %%
