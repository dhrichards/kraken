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
parser.add_argument("--lstar", type=float, default=0.005, help="Regularization length scale in meters")
parser.add_argument("--cellfactor", type=float, default=2, help="Mesh cell size factor")
parser.add_argument("--save_bp", type=bool, default=True, help="Save bp files")

args = parser.parse_args()


filename = "degradationtest_l" + str(args.lstar) + "_cellfactor" + str(args.cellfactor) + "_notdegraded"

path = './outputs'
os.makedirs(path, exist_ok=True)

nondim_length = 5.0

msh = kr.meshes.fenicsx_refined_mesh(nondim_length, args.lstar/args.cellfactor, 0.1, htop=0.2, large_size=0.2, top_fine_length=2.5, htop2 =1.1)
model = kr.base.Simulation(msh)

model.tol = 5e-6
model.min_its = 3
model.max_its = 400

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.T.value = -5.0
model.params.A0.value = mf.rate_factor_np(model.params.T.value)
model.params.H.value = 400
model.params.l.value = args.lstar*model.params.H.value
model.params.dt.value = 2.5*24*60*60
model.params.Kic.value = 100*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = -0.9*model.params.H.value
model.params.sea_level.value = 0.9 * model.params.H.value
model.params.length.value = nondim_length * model.params.H.value

model.params.σc = 200e3


if MPI.COMM_WORLD.rank == 0:
    print("ucstar: ", model.params.ucstar_float )
    print(path + "/" + filename)




def left_boundary(x):
    return np.isclose(x[0], 0)


u_bc = lambda V: [
                            bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                            bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                            ]


def fixed(x):
    return (x[0]<(nondim_length -0.3))*(x[1]<0.8) | (x[0]<(nondim_length - 2.0))


d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, end_cracks, 1.0),
                #   bc.internal_bc(V, lambda x: (x[1]>(1-height))*(1-end_cracks(x)), 0.0),
                ]


model.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])



t = 0.0
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)



model.damage_on = True



flag,nits = model.fixed_point(save=False, stop_bottom=True)

t += model.params.dt.value
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)

A = mf.rate_factor(model.params.T)/model.params.A0

η0 = mf.viscosity(ufl.dev(mf.ε(model.momentum.vel_prev_it)), 3.0, model.params.viscosity_tol, A=A)


    
kr.utilities.write_xdmf(path + "/" + filename + ".xdmf",
                        model.msh, [model.momentum.u,model.damage.d,model.damage.d_prev_it2,model.damage.d_prev_it,model.damage.d_prev_it3,
                                model.momentum.u_v, model.momentum.u_e,
                                model.momentum.ψplus/model.params.ψcritstar,
                                model.momentum.ε_e,
                                model.params.Gc,
                                η0,
                                model.params.ψcrit,
                                model.momentum.du,
                                model.momentum.du_smooth,
                                model.momentum.du_smooth-model.momentum.du,
                                ],
                                ["u","d","dprev2","dprev","dprev3",
                                "uv","ue",
                                "psi_plus",
                                "eps_e",
                                "Gc",
                                "eta",
                                "ψcrit",
                                "du",
                                "du_smooth",
                                "du_smooth_minus_du",
                                ],
                            t=0)

