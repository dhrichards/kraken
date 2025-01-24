#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
from kraken.material import MaterialProperties, Material_no_uc
from kraken.boundaryconditions import get_zero_bc
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.mainclass as mc

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

true_length = 4e3
true_height = 300




material = Material_no_uc()
material.L = true_height
material.τ = 3600*24  
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.l = 3.0/material.L


Hw = material.ρi/material.ρw*nondim_height


filename = "icebergL" + str(int(true_length/1e3)) + "l" + str(int(material.l*material.L)) + ".xdmf"

# msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
with io.XDMFFile(MPI.COMM_WORLD,"../meshes/" + filename,"r") as infile:
    msh = infile.read_mesh()

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
# cell_size = material.l/3
# nx = int(nondim_length/cell_size/2)
# nz = int(nondim_height/cell_size)

# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
#                             [nx,nz], mesh.CellType.quadrilateral)


#%%
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []

cliff_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type),
                        get_zero_bc(V.sub(1), bottom_boundary, default_scalar_type)]

model = mc.viscoelastic_damage(msh, [symm_bc,symm_bc,no_bc], material, 1.0)

g0 = 6.5
# g0=2.53
model.material.g = g0
steps = 20
#%%
for i in range(steps):
    model.material.g = g0 + i*(9.8-g0)/(steps-1)
    if MPI.COMM_WORLD.rank == 0:
        print(model.material.g)
   
    model.fixed_point(tol=1e-4,solve_stokes=False,max_its=100)
    ψp = mf.free_energy_plus(mf.ε(model.v),model.material.ν)
    pp = mf.positive_part(ψp-material.ψcritstar)
    pw = mf.water_pressure(msh,model.v)
    utilities.write_xdmf("/data/hpcdata/users/dancha/iceberginitial" + str(i) + ".xdmf",msh,\
                    [model.v,model.d,model.Hprev,pp],\
                    ["v","d","H","pp"],t=i)
    
# model.stokes.setup_solver(model.u,model.p,model.d,model.v)
# # model.material.g += (9.81-g0)/(steps-1)
for i in range(10):
    
    
    model.fixed_point(tol=1e-4,solve_stokes=False)
    # log.set_log_level(log.LogLevel.INFO)
    model.solve_stokes()
    

    utilities.write_xdmf("/data/hpcdata/users/dancha/iceberg" + str(i) + ".xdmf",msh,\
                    [model.v,model.d,model.u, mf.stress(model.v,material.ν)],\
                    ["v","d","u", "σ"],t=i)

    model.lagrangian_update()

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
