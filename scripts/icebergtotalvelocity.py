#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
from dolfinx import fem, io, log
import kraken as kr
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es

# log.set_log_level(log.LogLevel.INFO)

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
    return x[0]<(nondim_length/2 - 0.95*refineH[0]/params.L)#*(x[1]>-60))


## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 4e3
true_height = 300


path = './outputs'
os.makedirs(path, exist_ok=True)


params = kr.parameters.Params_with_uc()

# material = Material_with_uc()
params.L = true_height
params.l = 12
params.dt = 60*60*24
params.ψcrit = 0.5
params.Gc = 1.0
params.patm = 0.0

nondim_length = true_length/params.L
nondim_height = true_height/params.L
Hw = params.ρistar*nondim_height


refineH = (300*1.4,100)
msh = kr.utilities.create_refined_mesh(true_length, true_height, params,
                                     aspect_ratios=(100,1), refine=refineH,
                                     cell_factor=2.1)


no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc_mixed = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)
                        ]

model = kr.models.jakub2old.viscoelastic_damage(msh, [u_bc_mixed,bc_d], params)



#%%
min_its = 10
gs = [6.8,7.5,8.5]
gravfile = io.VTXWriter(MPI.COMM_WORLD,path + "/iceberggravity.bp", model.d)
gravfile.write(0)
for i in range(len(gs)):
    model.params.g = gs[i]
    model.setup()

    kr.iterators.fixed_point(model,min_its=min_its,tol=1e-4,max_its=200)

    gravfile.write(i+1)
    
    # model.d_prev_time.x.array[:] = model.d.x.array[:]
    

model.params.g = 9.8


model.setup()


functions = [model.u, model.d,
                    # model.u_v,model.u-model.u_v,
                    # model.u_v - model.u_v_prev_time,
                    model.Hprev,
                    es.free_energy_plus_spectral(model.ε_e, params.ν),
                    model.H]
names = ["u", "d",
                    #  "u_v","u_e",
                    #  "du_v",
                    "Hprev",
                    "psiplus",
                    "Hcurrent"
                    ]

vtx = kr.utilities.vtx_writer(path + "/test", msh, functions,names)

solve_damage = True
for i in range(300):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    if i > 20:
        min_its = 3
        solve_damage = True

    

    kr.iterators.fixed_point(model,
                             min_its=min_its,tol=1e-4,
                             solve_damage=solve_damage)#tol=-1, max_its = 10)


    vtx.write(functions,i)
    model.timestep()

    
