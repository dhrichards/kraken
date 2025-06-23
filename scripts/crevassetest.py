#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.parameters import Params_no_uc
import kraken.boundaryconditions as bc_bottom
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.mainclass as mc

def left_boundary(x):
    return np.isclose(x[0], -nondim_length/2)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], 0)

def crack(x):
    l = material.lstar
    return (x[0]>(-l/3))*(x[0]<(+l/3))*(x[1]>(nondim_height-l))

## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 2e3
true_height = 500

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



material = Params_no_uc()
material.L = true_height
material.τ = 3600*24
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.lstar = 20.0/material.L


cell_size = material.lstar/3.2

nx = int(nondim_length/cell_size)
nz = int(nondim_height/cell_size)

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([-nondim_length/2, 0]), np.array([nondim_length/2, nondim_height])],
                            [nx,nz], mesh.CellType.quadrilateral)

elast_bc = lambda V: [bc_bottom.get_bc(V.sub(0), left_boundary, -1e-6),
                        bc_bottom.get_bc(V.sub(0), right_boundary, 1e-6)]

no_bc = lambda V: []
bc_d = lambda V: [bc_bottom.internal_bc(V, crack, 1.0)]



model = mc.viscoelastic_damage(msh, [no_bc,no_bc,bc_d], material, 
                               dt = 1)#,g = lambda d: mf.degradation_Lo2023(d,0.05))

model.elastic.pw = lambda u: -1e-6
# # change w
# model.damage.w = lambda d: d
# model.damage.calc_c0()
# model.damage.bounded = True

#%%

for i in range(50):

    if MPI.COMM_WORLD.rank == 0:
        print(i)

    model.solve_damage()
    model.solve_elastic()

    utilities.write_xdmf(path + "crevassetest" + str(i) + ".xdmf",model.msh,
                        [model.v,model.d,model.u, mf.principal_stress(mf.ε(model.v),model.material.ν), mf.water_pressure(model.msh, model.v)],
                        ["v","d","u","λ","pw"],t=i)
    
