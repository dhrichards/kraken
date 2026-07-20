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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--l", type=float, default=2, help="Regularization length scale in meters")
parser.add_argument("--cellfactor", type=float, default=2.0, help="Mesh cell size factor")

args = parser.parse_args()


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], 0)

def top_boundary(x):
    return np.isclose(x[1], nondim_height)

def all_boundaries(x):
    return left_boundary(x) + right_boundary(x) + bottom_boundary(x) + top_boundary(x)
def crack(x):
    x_c = nondim_length/2 - 0.5*nondim_height
    width = args.l / (L)
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]<-0.5)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.9*nondim_height))# + (x[1]<(0.1-0.9*refineH[1]))


true_length = 5e3
true_height = 300




L = true_height
l = args.l


aspect_ratio_x = int(25/l)



nondim_length = true_length/L
nondim_height = true_height/L


refineH = (1.75,0.125)
# msh = kr.meshes.create_refined_mesh(nondim_length,nondim_height, l/L,
#                                      aspect_ratios=(aspect_ratio_x,1), refine=refineH,
#                                      cell_factor=args.cellfactor)

from dolfinx import mesh
cell_size = args.l/(args.cellfactor*L)
nx = int((nondim_length/2)/cell_size)
nz = int(nondim_height/cell_size)
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [nondim_length, nondim_height]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)

u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
d_bc = lambda V: [
    # bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, crack, 1.0)
                #   bc.get_zero_bc(V.sub(1), all_boundaries)
                  ]

model = kr.base.Simulation(msh)

model.params.H.value = true_height
model.params.l.value = args.l

model.params.Kic.value = 100e3
model.params.patm.value = 0.0
model.params.length.value = true_length
model.params.sea_level.value = model.params.H.value*0.9
# model.params.ge_tol.value = 1e-3

model.params.σt.value  = 200e3

model.setup(kr.momentum.elastic.Elasticity,kr.damage.higherorder.AT2,[u_bc, d_bc])


#%%

model.damage_on = True
model.fixed_point(save=True)


kr.utilities.write_xdmf("./outputs/elastictestx" + args.type + ".xdmf",
                            msh, [model.momentum.u, model.damage.d, 
                                  model.momentum.ψplus,
                                #   ψplus,
                                #   ψplus2,
                                    model.momentum.ε_e,
                                #   ufl.tr(model.momentum.ε_eD),
                                #   es.free_energy_plus_lo_pressure(model.momentum.ε_e, model.momentum.p, model.params.ν),
                                  ],
                            ["u", "d", 
                             "psi_plus",
                            #  "psi_plus_2",
                             "e_e",
                            #  "tr_e_d",
                            #  "psi_plus_pressure"
                             ])
    # model.d_prev_time.x.array[:] = model.d.x.array[:]

