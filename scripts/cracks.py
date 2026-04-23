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
params.H = 300.00
params.l = 12.0
params.dt = 60*60*2
params.ψcrit = 0.0
params.Gc = 1.0
params.patm = 0.0

nondim_length = true_length/params.H
nondim_height = true_height/params.H
Hw = params.ρistar*nondim_height


refineH = (1.4,0.3)
msh = kr.utilities.create_refined_mesh(true_length, true_height, params,
                                     aspect_ratios=(100,1), refine=refineH,
                                     cell_factor=2.1)
# msh.geometry.x[:,1] += 0.5

no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

model = kr.models.jakub2.viscoelastic_damage(msh, [u_bc,bc_d], params)

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%
min_its = 10


# H = mf.clayton_driving_function(es.cauchy_stress(model.ε_e, model.params.ν), model.params.σcritstar, mf.water_pressure_static(model.msh))

gs = [4,6.8,7.5,8.5]


for i in range(len(gs)):
    model.params.g = gs[i]
    model.setup_all()

    kr.iterators.fixed_point(model,min_its=min_its)

    kr.utilities.write_xdmf(path + "/iceberggravity" + str(i) + ".xdmf",
                            msh, [model.u,model.d,es.free_energy_plus_dp(model.ε_e,model.params.ν)],
                            ["u","d","pp"], t=i)
    

model.params.g = 9.8
model.setup_all()




for i in range(300):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    if i > 20:
        min_its = 3

    kr.iterators.fixed_point(model,min_its=min_its,tol=1e-5,solve_damage=True)#tol=-1, max_its = 10)

    kr.utilities.write_xdmf(path + "/iceberg" + str(i) + ".xdmf",
                            msh, [model.u,model.d,es.free_energy_plus_dp(model.ε_e,params.ν)],["u","d","pp"], t=i)

    model.timestep()
   
