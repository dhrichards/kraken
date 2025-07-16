#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
from dolfinx import io
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def crack(x):
    x_c = nondim_length/2 - nondim_height
    l = params.lstar
    return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return x[0]<(nondim_length/2 - refineH[0]*0.9*nondim_height)#*(x[1]>-60))


## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 16e3
true_height = 300


path = './outputs'
os.makedirs(path, exist_ok=True)


params = kp.Params_with_uc()

# material = Material_with_uc()
params.L = 300.00
params.l = 30
params.dt = 60*60*2
params.ψcrit = 0.5
params.Gc = 1.0
params.patm = 0.0

nondim_length = true_length/params.L
nondim_height = true_height/params.L
Hw = params.ρistar*nondim_height


refineH = (2.5,0.3)
msh = kr.utilities.create_refined_mesh(16e3, 300, params,
                                     aspect_ratios=(30,1), refine=refineH,
                                     cell_factor=2.1)


no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc_mixed = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)]

u_bc_2 = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
models = []

# models.append(kr.models.jakub2.viscoelastic_damage(msh, [u_bc_mixed,bc_d], params))
# models.append(kr.models.jakub3.viscoelastic_damage(msh, [u_bc_mixed,bc_d], params))
models.append(kr.models.stokes.viscoelastic_damage(msh, [u_bc_2,bc_d], params))


#%%
min_its = 5


functions = []
names = []
for model in models:
    functions.append(model.u)
    # functions.append(model.ε_e)


for i in range(len(functions)):
    names.append("u" + (str(i+1)))
    # names.append("eps" + (str(i+1)))





for model in models:
    model.setup()





vtx = kr.utilities.vtx_writer(path + "/viscoelasticcomparison", msh, functions,names)

for i in range(50):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    for model in models:
        kr.iterators.fixed_point(model,min_its=min_its,tol=1e-5,solve_damage=False)#tol=-1, max_its = 10)

    vtx.write(functions,i)

    for model in models:
        model.timestep()

