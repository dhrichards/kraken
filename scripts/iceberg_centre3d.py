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


def west_boundary(x):
    return np.isclose(x[0], 0)

def east_boundary(x):
    return np.isclose(x[0], nondim_length_x)

def south_boundary(x):
    return np.isclose(x[1], 0)

def north_boundary(x):
    return np.isclose(x[1], nondim_length_x)

def bottom_boundary(x):
    return np.isclose(x[2], 0)

def top_boundary(x):
    return np.isclose(x[2], nondim_height)


def crack(x):
    return (x[0]<nondim_length_x/2+1e-3)*(np.abs(x[1]-nondim_length_x/2)<0.51*h/L)




true_length_x = 200
true_length_y = 100
true_height = 100

L = true_height
l = 5.0
ρi = 900
ρsw = 1000


# path = '/data/hpcdata/users/dancha/outputs'
path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length_x = true_length_x/L
nondim_length_y = true_length_y/L
nondim_height = true_height/L

cell_factor = 1
h = l/cell_factor
msh = mesh.create_box(MPI.COMM_WORLD,
                      [[0.0, 0.0, 0.0], [nondim_length_x, nondim_length_y, nondim_height]],
                      [int(true_length_x/h), int(true_length_y/h), int(true_height/h)],
                      cell_type=mesh.CellType.hexahedron)



# d_bc = lambda V: [bc.internal_bc(V, crack, 1.0)]
d_bc = lambda V: []


u_bc = lambda V: [  
            bc.get_bc(V.sub(1), north_boundary, 0.01),
            bc.get_zero_bc(V.sub(1), south_boundary),
            bc.get_zero_bc(V.sub(0), west_boundary),
            bc.get_zero_bc(V.sub(0), east_boundary),
            bc.get_zero_bc(V.sub(2), bottom_boundary),
            
            ]



model = kr.base.Simulation(msh, [u_bc, d_bc],
                           kr.momentum.elastic.ElasticEnergySplit,
                           kr.damage.higherorder.HigherOrder, split="lo_3d")


# model.T = mf.temperature(msh,ρi/ρsw,-30,-2)
model.params.H.value = L
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

load_steps = np.linspace(0.0,0.1,20)

for i in range(len(load_steps)):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    u_bc = lambda V: [  
            bc.get_bc(V.sub(1), north_boundary, load_steps[i]),
            bc.get_zero_bc(V.sub(1), south_boundary),
            bc.get_zero_bc(V.sub(0), west_boundary),
            bc.get_zero_bc(V.sub(0), east_boundary),
            bc.get_zero_bc(V.sub(2), bottom_boundary),
            
            ]

    model.momentum.bc_u = u_bc(model.momentum.U)
    model.momentum.setup()


    model.fixed_point(min_its=3, tol=1e-5, max_its=150, solve_damage=True)


    kr.utilities.write_xdmf(path + "/3ddirect_run" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                    #   model.momentum.u_e, model.momentum.u_v,
                                    ],
                                    ["u","d",
                                    "ue","uv"
                                    ],
                                  t=i)

    model.damage.timestep()



   
