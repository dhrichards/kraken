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

# def bottom_boundary(x):
#     return np.isclose(x[1], -Hw)

# def crack(x):
#     x_c = nondim_length/2 - nondim_height
#     l = params.lstar
#     return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.9*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))

true_length = 16e3
true_height = 300


path = './outputs'
os.makedirs(path, exist_ok=True)


L = true_height
l = 5.0


nondim_length = true_length/L
nondim_height = true_height/L


refineH = (2.5,0.3)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L,
                                     aspect_ratios=(100,1), refine=refineH,
                                     cell_factor=1)
# msh.geometry.x[:,1] += 0.5

no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V.sub(0), fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

model = kr.models.jakub3.viscoelastic_damage(msh, [u_bc,bc_d])

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))
model.params.L.value = L
model.params.l.value = l
model.params.dt.value = 60*60*24
model.params.ψcrit.value = 1.0
model.params.Gc.value = 1.0
model.params.patm.value = 0.0

#%%
min_its = 4
model.setup_all()

solve_d = False


for i in range(300):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    if i == 10:
        solve_d = True
        # model.params.dt = 60*60*12
        # model.setup_all()

    if i == 20:
        min_its = 3

    kr.iterators.fixed_point(model,min_its=min_its,tol=1e-5,solve_damage=solve_d,max_its=300)#tol=-1, max_its = 10)

    kr.utilities.write_xdmf(path + "/relax" + str(i) + ".xdmf",
                            msh, [model.u,model.d,es.free_energy_plus_dp(model.ε_e,model.params.ν.value)],["u","d","pp"], t=i)

    model.timestep()
   
