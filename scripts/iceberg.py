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
    return (x[0]<(nondim_length/2 - refineH[0]*0.98*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))



true_length = 16e3
true_height = 300

L = true_height
l = 6
ρi = 900
ρf = 350
ρsw = 1000
D = 32.5


path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = true_length/L
nondim_height = true_height/L

# flotation_height = mf.flotation_height(ρi/ρsw,ρf/ρsw,D/L)
flotation_height = ρi/ρsw

refineH = (2.5,0.4)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L, flotation_height,
                                     aspect_ratios=(300,1), refine=refineH,
                                     cell_factor=1.3)


d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        #    bc.get_zero_bc(V.sub(1).sub(0), left_boundary)
                        ]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

model = kr.base.Simulation(msh, [u_bc, d_bc],
                           kr.momentum.mixed.SemiLagrangian,
                           kr.damage.higherorder.HigherOrder, 0.06)



model.params.L.value = L
model.params.l.value = l
model.params.dt.value = 60*60*24*3
model.params.ρi.value = ρi
model.params.ρw.value = ρsw
model.params.ψcrit.value = 1.0
model.params.Gc.value = 1.0
model.params.patm.value = 0.0


#%%
min_its = 10

# model.setup_all()
model.setup()

gs = [6,8,9]

for i,g in enumerate(gs):

    model.params.g.value = g

    model.fixed_point(min_its=min_its, tol=1e-5,max_its=50)

    kr.utilities.write_xdmf(path + "/iceberggravity" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                #   model.momentum.u_e, model.momentum.u_v,
                                ],
                                  ["u","d",
                                "ue","uv"
                                ],
                                  t=i)
    model.damage.timestep()
    # model.d_prev_time.x.array[:] = model.d.x.array[:]
    
#%%

model.params.g.value = 9.8

from dolfinx import fem
import ufl
min_its = 5

for i in range(300):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    
    if i == 5:
        min_its = 3

    model.fixed_point(min_its=min_its, tol=1e-5)



    kr.utilities.write_xdmf(path + "/icebergold" + str(i) + ".xdmf",
                            msh, [model.momentum.u, model.damage.d,model.momentum.ρ,
                                #   model.momentum.u_e, model.momentum.u_v,
                                #   ufl.div(model.momentum.vel),ufl.div(model.momentum.du_e),
                                  ],
                                  ["u", "d","ρ",
                                "ue", "uv",
                                # "div_uv","div_ue"
                                ],
                                  t=i)

    model.timestep()
   
