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

# def crack(x):
#     x_c = nondim_length/2 - nondim_height
#     l = params.lstar
#     return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.9*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))



true_length = 16e3
true_height = 300

L = true_height
l = 20.0


path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = true_length/L
nondim_height = true_height/L

refineH = (2.0,0.3)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L,
                                     aspect_ratios=(100,1), refine=refineH,
                                     cell_factor=2.1)
# msh.geometry.x[:,1] += 0.5

no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

model = kr.models.jakub3.viscoelastic_damage(msh, [u_bc,bc_d])


model.params.L.value = L
model.params.l.value = l
model.params.dt.value = 60*60*24
model.params.ψcrit.value = 0.0
model.params.Gc.value = 1.0
model.params.patm.value = 0.0

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%
min_its = 2

model.setup_all()
gs = [2,3,4,5,6,7,8,9]

for i,g in enumerate(gs):

    model.params.g.value = g

    print(model.params.ucstar_float)

    kr.iterators.fixed_point(model, min_its=min_its, tol=1e-7,max_its=4)

    kr.utilities.write_xdmf(path + "/iceberggravity" + str(i) + ".xdmf",
                            msh, [model.u,model.d,
                                  model.u_e, model.u_v,],
                                  ["u","d",
                                "ue","uv"],
                                  t=i)
    
    # model.d_prev_time.x.array[:] = model.d.x.array[:]
    
    

model.params.g.value = 9.8

min_its = 5

for i in range(300):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    
    if i == 20:
        min_its = 3

    kr.iterators.fixed_point(model,min_its=min_its,tol=1e-5)#tol=-1, max_its = 10)

    kr.utilities.write_xdmf(path + "/iceberg" + str(i) + ".xdmf",
                            msh, [model.u,model.d],["u","d"], t=i)

    model.timestep()
   
