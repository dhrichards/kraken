#%%
from mpi4py import MPI
import numpy as np
from dolfinx import mesh
import kraken as kr
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf

dt_days = 2.5
length = 2.5e3
height = 450
l = 8
small_size = l/2
refinements = 4

large_size = small_size*2**refinements

nx = int(length/large_size)
nz = int(height/large_size)
nondim_length = length/height
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [nondim_length, 1.0]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)

def refined_area(x):
    return (x[0] > 0.95*nondim_length)\
                |((x[1]>0.85)*(x[0]>0.6*nondim_length))

msh = kr.meshes.refine_by_area(msh,refined_area,refinements)

model = kr.base.Simulation(msh)

model.params.T.value = -5.0
model.params.A0.value = mf.rate_factor_np(-5)
model.params.H.value = height
model.params.l.value = l
model.params.dt.value = dt_days*24*60*60


def left_boundary(x):
    return np.isclose(x[0], 0)

u_bc = lambda V: [
                            bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                            bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                            ]

d_bc = lambda V: []


model.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])

kr.plotting.write_xdmf("test.xdmf",msh,[model.momentum.u],["u"])


model.damage_on = False


relax_time_days = 400
nt = 10
model.params.dt.value = 400*24*60*60 / nt
for i in range(nt):
    if MPI.COMM_WORLD.rank == 0:
        print("Relaxation iteration: ", i)
    flag,nits = model.fixed_point(save=True)
    if flag == -1:
        break
    model.momentum.timestep()

model.params.dt.value = dt_days*24*60*60



t = 0.0
model.damage_on = True

for i in range(1,500):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    flag,nits = model.fixed_point(stop_bottom=True)

    t += model.params.dt.value
    kr.plotting.write_xdmf("iceshelf_step" + str(i) + ".xdmf",
                            model.msh, [model.momentum.u,model.damage.d,
                                    model.momentum.u_v, model.momentum.u_e,
                                    model.momentum.ψplus/model.params.ψcritstar,
                                    ],
                                    ["u","d",
                                    "uv","ue",
                                    "psi_plus",
                                    ],
                                t=i)
    
    if flag == -1:
        break

    
    model.timestep()
    # model.momentum.timestep()


