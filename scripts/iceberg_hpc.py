#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
from dolfinx import fem
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken.utilities as utilities
import kraken.total_velocity as tv
import kraken.total_displacement as td
import kraken.jakub as jk
import kraken.jakub2 as jk2
import kraken.jakub3 as jk3
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
    return x[0]<(nondim_length/2 - 0.95*refineH[0]/params.L)#*(x[1]>-60))


## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 800
true_height = 300


path = './outputs'
os.makedirs(path, exist_ok=True)


params = kp.Params_with_uc()

# material = Material_with_uc()
params.L = true_height
params.l = 2.0
params.dt = 60*60*24
params.ψcrit = 1.0
params.Gc = 1.0
params.patm = 1e5
params.crack_viscosity = 0.0

nondim_length = true_length/params.L
nondim_height = true_height/params.L
Hw = params.ρistar*nondim_height


refineH = (600,50)
msh = utilities.create_refined_mesh(true_length, true_height, params,
                                     aspect_ratios=(1,50), refine=refineH,
                                     cell_factor=2.1)


no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc_mixed = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)
                        ]

model = jk2.viscoelastic_damage(msh, [u_bc_mixed,no_bc], params)


# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))

#%%
min_its = 20
# gs = [9.5]

# for i in range(len(gs)):
#     model.params.g = gs[i]
#     model.setup_displacement()
#     model.setup_damage() 

#     model.fixed_point(min_its=min_its,tol=1e-4,max_its=200)

#     utilities.write_xdmf(path + "/iceberggravity" + str(i) + ".xdmf", msh,
#                         [model.u, model.d, 
#                         #  model.u_v,model.u-model.u_v
#                          ],
#                     ["u", "d",
#                     #   "u_v","u_e"
#                       ], t=i)
    
#     # model.d_prev_time.x.array[:] = model.d.x.array[:]
    

# model.params.g = 9.8


model.setup_displacement()
model.setup_damage()

solve_damage = True
for i in range(3):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    if i > 5:
        min_its = 3
        solve_damage = True

    

    model.fixed_point(min_its=min_its,tol=1e-4,solve_damage=solve_damage)#tol=-1, max_its = 10)

    p_ext = mf.water_pressure(model.msh,model.u,model.params.ucstar) +model.params.patmstar
    
    ψ = es.free_energy(model.ε_e, model.params.ν)
    ψplus = es.free_energy_plus_spectral(model.ε_e, params.ν)
    ψminus = ψ - ψplus

    H2 = mf.history_function(model.ε_e, 0.0, model.params.ν, model.params.ψcritstar)

    utilities.write_xdmf(path + "/fine" + str(i) + ".xdmf", msh,
                        [model.u, model.d,
                        # model.u_v,model.u-model.u_v,
                        model.Hprev, H2, ψplus, ψminus,
                         p_ext*ufl.grad(model.g)],
                    ["u", "d", 
                    #  "u_v","u_e",
                     "Hprev","H2","psiplus","psiminus",
                     "test"], t=i)
    # model.move_mesh()

    model.timestep()

    
