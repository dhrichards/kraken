#%%
import numpy as np
from dolfinx import mesh, default_scalar_type
from mpi4py import MPI
import numpy as np
from kraken.parameters import Params_no_uc
import kraken.boundaryconditions as bc_bottom
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.mainclass as mc

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def top_boundary(x):
    return np.isclose(x[1], nondim_height-Hw)



true_length = 4e3
true_height = 300




material = Params_no_uc()
material.L = true_height
material.τ = 3600*24  
nondim_length = true_length/material.L
nondim_height = true_height/material.L

Hw = material.ρi/material.ρw*nondim_height

# cell_size = material.l/3
nx = 50
nz = 10

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length, nondim_height-Hw])],
                            [nx,nz], mesh.CellType.quadrilateral)


clamped_bc = lambda V: [bc_bottom.get_zero_bc(V, left_boundary)]
symm_bc = lambda V: [bc_bottom.get_zero_bc(V.sub(0), left_boundary)]
no_bc = lambda V: []

z_bc = lambda V: [bc_bottom.get_bc(V, bottom_boundary, default_scalar_type(-Hw))]


u_in_bc = lambda V: [bc_bottom.get_bc(V, left_boundary, np.array([1.0,0.0]))]


model = mc.viscoelastic_damage(msh, [clamped_bc,u_in_bc,no_bc,no_bc], material,
                                1.0,
                               lambda x: top_boundary(x) + bottom_boundary(x))


#%%

for i in range(1000):
    
    # log.set_log_level(log.LogLevel.INFO)
    # model.solve_elastic()
    model.solve_stokes()
    model.eulerian_update()

    utilities.write_xdmf("outputs/shelf" + str(i) + ".xdmf",msh,\
                    [model.v,model.d,model.u, mf.cauchy_stress(model.v,material.ν)],\
                    ["v","d","u", "σ"],t=i)

    
