#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
import dolfinx
from dolfinx import io, mesh
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lstar", type=float, default=0.06, help="Regularization length scale in meters")
parser.add_argument("--cellfactor", type=float, default=2, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=500, help="Height of iceberg in meters")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--save_bp", type=bool, default=False, help="Save bp files")
args = parser.parse_args()


filename = "ssa_H" + str(args.height) \
                        + "_l" + str(args.lstar) \
                    + "_cellfactor" + str(args.cellfactor)\
                            + "_" + args.suffix + "_"



def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length)


path = './outputs'
os.makedirs(path, exist_ok=True)




nondim_length = 2
nondim_height = 1.0


# cell_size = 0.2 # large size
cell_size = args.lstar/args.cellfactor
nx = int((nondim_length)/cell_size)
if np.mod(nx,2) == 0:
    nx = nx + 1

nz = int(nondim_height/cell_size)
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [nondim_length, nondim_height]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)


model = kr.base.Simulation(msh)

model.tol = 5e-6
model.min_its = 200
model.max_its = 1300

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.T.value = -5
model.params.A0.value = mf.rate_factor_np(model.params.T.value)
model.params.H.value = args.height
model.params.l.value = args.lstar*args.height
model.params.dt.value = 2.5*24*60*60
model.params.Kic.value = 100*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = 0.0
model.params.sea_level.value = 0.9 * args.height
model.params.length.value = nondim_length * args.height
model.params.σt.value = 200e3


if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)



δ = model.params.δ; ν = model.params.ν
edotvxx = (δ/4)**3
duv = lambda x: edotvxx*x[0]*model.params.dtstar

duedx = lambda z: (-0.125*δ*ν + 0.25*δ + 1.0*ν - 0.5 - 1.0*z*ν + 0.5*z)/((ν + 1))
ue_x = lambda x: duedx(x[1])*x[0]
dudx = model.params.dtstar*(δ/4)**3
u_x = lambda x: dudx*x[0]
uv_x = lambda x: u_x(x) - ue_x(x)
u_bc = lambda V: [
                    bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                    bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                    bc.get_bc_func(V.sub(1).sub(0), right_boundary, uv_x),
                    bc.get_bc_func(V.sub(0).sub(0), right_boundary, u_x),
                    # bc.get_bc_func(V.sub(0).sub(1),left_boundary, lambda x: -dudx*x[1]),
                    # bc.get_bc_func(V.sub(0).sub(1),right_boundary, lambda x: -dudx*x[1]),
                    # bc.get_bc(V.sub(0).sub(1), bottom_boundary, 0.0),
                    ]


d_bc = lambda V: []


model.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])




t = 0.0
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)


model.damage_on = True




flag,nits = model.fixed_point(save=True)

t += model.params.dt.value
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)
    
kr.utilities.write_xdmf(path + "/" + filename +".xdmf",
                                model.msh, [model.momentum.u,model.damage.d,model.damage.d_prev_it2,model.damage.d_prev_it,model.damage.d_prev_it3,
                                        model.momentum.u_v, model.momentum.u_e,
                                        model.momentum.ψplus/model.params.ψcritstar,
                                        model.momentum.ε_e,
                                        model.params.Gc,
                                        model.params.ψcrit,
                                        model.momentum.du,
                                        ],
                                        ["u","d","dprev2","dprev","dprev3",
                                        "uv","ue",
                                        "psi_plus",
                                        "eps_e",
                                        "Gc",
                                        "ψcrit",
                                        "du",
                                        ],
                                    t=1)
    
 
if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)


   
