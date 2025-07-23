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
params.l = 18.0
params.dt = 60*60*2
params.ψcrit = 0.5
params.Gc = 1.0
params.patm = 0.0

nondim_length = true_length/params.L
nondim_height = true_height/params.L
Hw = params.ρistar*nondim_height


refineH = (2.5,0.3)
msh = kr.utilities.create_refined_mesh(16e3, 300, params,
                                     aspect_ratios=(100,1), refine=refineH,
                                     cell_factor=2.1)


no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc_mixed = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1), left_boundary)]

model = kr.models.total_displacement.viscoelastic_damage(msh, [u_bc_mixed,bc_d], params)

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%
min_its = 1



H = mf.clayton_driving_function(es.cauchy_stress(model.ε_e, model.params.ν), model.params.σcritstar, mf.water_pressure_static(model.msh))

functions = [model.u, model.d,
                    # model.u_v,model.u-model.u_v,
                    # model.u_v - model.u_v_prev_time,
                    # model.u_prev_time, model.u_v_prev_time,
                    model.Hprev,
                    es.free_energy_plus_spectral(model.ε_e, params.ν),
                    H
                   ]
names = ["u", "d",
                    #  "u_v","u_e",
                    #  "du_v",
                    # "u_prev_time", "u_v_prev_time",
                    "Hprev",
                    "psiplus",
                    "H"
                    ]

# gs = [6.8,7.5,8.5,9.2]
# gravfile = kr.utilities.vtx_writer(path + "/iceberggravity", msh, functions,names)
# gravfile.write(functions,0)

# for i in range(len(gs)):
#     model.params.g = gs[i]
#     model.setup()

#     kr.iterators.fixed_point(model,min_its=10)

#     gravfile.write(functions,i+1)
    

# model.params.g = 9.8
model.setup()




vtx = kr.utilities.vtx_writer(path + "/totaldisp", msh, functions,names)

for i in range(50):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    if i > 20:
        min_its = 3

    kr.iterators.fixed_point(model,min_its=min_its,tol=1e-5,solve_damage=False)#tol=-1, max_its = 10)

    vtx.write(functions,i)

    model.timestep()
    # model.w_prev_time.x.array[:] = model.w.x.array[:]
    # model.d_lb.x.array[:] = model.d.x.array[:]
