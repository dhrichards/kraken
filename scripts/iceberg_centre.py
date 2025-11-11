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




true_length = 200
true_height = 100

L = true_height
l = 5.0
ρi = 900
ρsw = 1000


# path = '/data/hpcdata/users/dancha/outputs'
path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = true_length/L
nondim_height = true_height/L


refineH = (2.5,0.4)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L, 0.0,
                                     aspect_ratios=(1,1), refine=refineH,
                                     cell_factor=1, cell_type=mesh.CellType.quadrilateral)


d_bc = lambda V: []

u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                    bc.get_zero_bc(V.sub(1), bottom_boundary),
                    bc.get_bc(V.sub(0), right_boundary, 0.01),
                    ]



model = kr.base.Simulation(msh, [u_bc, d_bc],
                           kr.momentum.elastic.ElasticEnergySplit,
                           kr.damage.higherorder.HigherOrder, split="lo")


# model.T = mf.temperature(msh,ρi/ρsw,-30,-2)
model.params.L.value = L
model.params.l.value = l
model.params.dt.value = 1*24*60*60
model.params.ρi.value = ρi
model.params.ρw.value = ρsw
# model.params.ρc = model.params.ρi
model.params.ψcrit.value = 1.0
model.params.Gc.value = 0.5
model.params.patm.value = 0.0


#%%

model.setup()

load_steps = np.linspace(0.01,1.0,50)

for i in range(len(load_steps)):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1), bottom_boundary),
                            bc.get_bc(V.sub(0), right_boundary, load_steps[i]),
                            ]
    model.momentum.bc_u = u_bc(model.momentum.U)
    model.momentum.setup()


    model.fixed_point(min_its=3, tol=1e-5, max_its=150, solve_damage=True)


    kr.utilities.write_xdmf(path + "/centretest_run" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                    #   model.momentum.u_e, model.momentum.u_v,
                                    ],
                                    ["u","d",
                                    "ue","uv"
                                    ],
                                  t=i)

    # model.damage.timestep()



   
