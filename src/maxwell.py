#%%

from dolfinx import mesh, io, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
from kraken.material import Material_no_uc
from kraken.boundaryconditions import get_zero_bc
from kraken import bodyforces as bf, elasticityclass as ec, phasefield as pf, meshes, utilities
import gmsh


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def notch(x):
    xc = nondim_length/2 - nondim_height*0.7
    w = material.l*2.5
    d = (nondim_height-Hw)*-0.7
    return (x[0]>xc-w/2)*(x[0]<xc+w/2)*(x[1]>d)

def bc_notch(V):
    deactivate_cells = mesh.locate_entities(msh, msh.topology.dim, notch)
    deactivate_dofs = fem.locate_dofs_topological(V, msh.topology.dim, deactivate_cells)
    return [fem.dirichletbc(default_scalar_type(1.0), deactivate_dofs, V)]

true_length = 4e3
true_height = 300



material = Material_no_uc()
material.L = true_height
material.τ = 3600*24
# material.L = true_height    
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.l = 2.0/material.L


Hw = material.ρi/material.ρw*nondim_height
# true_water_depth = 100
# Hw = true_water_depth/material.L

model = meshes.create_iceberg_mesh(true_length, true_height, material)
msh,ct,ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

gmsh.finalize()
# cell_size = material.l/3
# nx = int(nondim_length/cell_size/2)
# nz = int(nondim_height/cell_size)

# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
#                             [nx,nz], mesh.CellType.quadrilateral)

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
#%%
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []

cliff_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type),
                        get_zero_bc(V.sub(1), bottom_boundary, default_scalar_type)]

model = ec.viscoelastic_damage(msh, [symm_bc,symm_bc,no_bc], material, 1.0)

g0 = 6.5
# g0=2.53
model.material.g = g0
steps = 20
for i in range(steps):
    model.material.g = g0 + i*(9.8-g0)/(steps-1)
    if MPI.COMM_WORLD.rank == 0:
        print(model.material.g)
   
    model.fixed_point(tol=1e-4,solve_stokes=False,max_its=100)
    ψp = pf.free_energy_plus(pf.ε(model.v),model.material.ν)
    pp = pf.positive_part(ψp-material.ψcritstar)
    pw = bf.water_pressure(msh,model.v)
    utilities.write_xdmf("outputs/iceberginitial" + str(i) + ".xdmf", msh, \
                         [model.v,model.d,model.Hprev,pp], \
                         ["v","d","H","pp"], t=i)
    

# # model.material.g += (9.81-g0)/(steps-1)
for i in range(1000):
    
    
    model.fixed_point(tol=1e-4,solve_stokes=False)
    # log.set_log_level(log.LogLevel.INFO)
    model.solve_stokes()
    

    utilities.write_xdmf("outputs/iceberg" + str(i) + ".xdmf", msh, \
                         [model.v,model.d,model.u, pf.stress(model.v,material.ν)], \
                         ["v","d","u", "σ"], t=i)

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

# utilities.write_vtk("outputs/pf.pvd",msh,\
#                     [model.u,model.v,model.d],\
#                     ["u","v","d"],t=0)



# %%
