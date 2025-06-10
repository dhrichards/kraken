#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken.utilities as utilities
import kraken.total_velocity as tv
import kraken.total_displacement as td
import kraken.jakub as jk
import kraken.jakub2 as jk2
import kraken.oneclass as oc
import kraken.numerics.total_velocity_maths as tvm

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
    return x[0]<(nondim_length/2 - 1.4*nondim_height)#*(x[1]>-60))


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
params.l = 10.0
params.dt = 60*60*24*2
params.ψcrit = 1.5
params.patm = 0.0

nondim_length = true_length/params.L
nondim_height = true_height/params.L
Hw = params.ρistar*nondim_height



msh = utilities.create_refined_mesh(16e3, 300, params,
                                     aspect_ratios=(100,100), refineH=(1.5,0.3),
                                     cell_factor=2.1)


no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc_mixed = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)]

model = jk2.viscoelastic_damage(msh, [u_bc_mixed,bc_d], params)


# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%
model.setup_displacement()
model.setup_damage()


for i in range(300):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    model.fixed_point(min_its=10)#tol=-1, max_its = 10)

    p_ext = mf.water_pressure(model.msh,model.u,model.params.uc_star) +model.params.patmstar
    
    ψplus = es.free_energy_plus_spectral(mf.ε(model.u_e), params.ν)
    utilities.write_xdmf("path/iceberginit" + str(i) + ".xdmf", msh,
                        [model.u, model.d, model.u_v,
                         model.u-model.u_v,model.Hprev, 
                         ufl.div(model.u_v-model.u_v_prev_time), ψplus,
                         p_ext*ufl.grad(model.g)],
                    ["u", "d", "u_v","u_e","Hprev","div_u_v","psiplus","test"], t=i)
    # model.move_mesh()
    model.w_prev_time.x.array[:] = model.w.x.array[:]
