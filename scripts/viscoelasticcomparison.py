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
params.l = 100
params.dt = 60*60*24
params.ψcrit = 0.5
params.Gc = 1.0
params.patm = 0.0

nondim_length = true_length/params.L
nondim_height = true_height/params.L
Hw = params.ρistar*nondim_height


refineH = (2.5,0.3)
msh = kr.utilities.create_refined_mesh(16e3, 300, params,
                                     aspect_ratios=(1,1), refine=refineH,
                                     cell_factor=2.1)


no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
models = []

models.append(kr.models.jakub2.viscoelastic_damage(msh, [u_bc,bc_d], params))
# models.append(kr.models.jakub3.viscoelastic_damage(msh, [u_bc_mixed,bc_d], params))
models.append(kr.models.jakub3.viscoelastic_damage(msh, [u_bc,bc_d], params))


#%%
min_its = 10




for model in models:
    model.setup_all()



for i in range(500):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)



    for model in models:
        kr.iterators.fixed_point(model,min_its=min_its,tol=1e-5,solve_damage=False)#tol=-1, max_its = 10)

    kr.utilities.write_xdmf(path + "/viscoelasticcomparison" + str(i) + ".xdmf", msh, 
                            [models[0].u,models[1].u,
                                models[0].p,models[1].p,
                                es.free_energy_plus_spectral(models[0].ε_e,params.ν),
                                es.free_energy_plus_spectral(models[1].ε_e,params.ν)],
                                ["u1","u2",
                                 "p1","p2",
                                 "f1","f2"], t=i)
    for model in models:
        model.timestep()
        # model.p_prev_time.x.array[:] = model.p.x.array[:]

