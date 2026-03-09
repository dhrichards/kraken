#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
from dolfinx import io, mesh, default_scalar_type
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
    return np.isclose(x[1], 0)

H = 300
true_length = H*4
lstar = 0.01
l = H*lstar

nondim_length = true_length/H


cell_factor = 1.0

cell_size = l/H/cell_factor
nx = int(nondim_length/cell_size/2)
nz = int(1/cell_size)
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [nondim_length/2, 1]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)


model = kr.base.Simulation(msh,split="lo_p")


# model.T = mf.temperature(msh,ρi/ρsw,-30,-2)
model.params.H.value = H
model.params.l.value = l
model.params.ψcrit.value = 0.5
model.params.Gc.value = 1.0
model.params.patm.value = 0.0
model.params.length.value = true_length
model.params.sea_level.value = 0.9*model.params.H.value


δ = model.params.δ; ν = model.params.ν
exx = lambda z: (-0.125*δ*ν + 0.25*δ + 1.0*ν - 0.5 - 1.0*z*ν + 0.5*z)/((ν + 1))


u_x = lambda x: exx(x[1])*nondim_length/2

path = './outputs'
os.makedirs(path, exist_ok=True)



d_bc = lambda V: []

u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                    # bc.get_zero_bc(V.sub(1), bottom_boundary),
                    bc.get_bc_func(V.sub(0), right_boundary, u_x),
                    ]





#%%

model.setup(MomentumSolver=kr.momentum.elastic.Elasticity,bc_funcs = [u_bc,d_bc])

# load_steps = np.linspace(0.01,1.0,50)

# for i in range(len(load_steps)):

#     if MPI.COMM_WORLD.rank == 0:
#         print("Iteration: ", i)

#     u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
#                            bc.get_zero_bc(V.sub(1), bottom_boundary),
#                             bc.get_bc(V.sub(0), right_boundary, load_steps[i]),
#                             ]
#     model.momentum.bc_u = u_bc(model.momentum.U)
#     model.momentum.setup()


#     model.fixed_point(min_its=3, tol=1e-5, max_its=150, solve_damage=True)

# model.momentum.solve()
model.damage_on = True
model.fixed_point(save=True)

kr.utilities.write_xdmf(path + "/centretest_run.xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                  model.momentum.ψplus/model.params.ψcritstar,
                                    #   model.momentum.u_e, model.momentum.u_v,
                                    ],
                                    ["u","d",
                                     "psiplus"
                                    "ue","uv"
                                    ],
                                  t=0)

    # model.damage.timestep()



   
