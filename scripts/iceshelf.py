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
parser.add_argument("--level", type=float, default=0.00, help="Water level in cracks above sea level (m)")
parser.add_argument("--lstar", type=float, default=0.005, help="Regularization length scale in meters")
parser.add_argument("--dt", type=float, default=2.5, help="Time step in days")
parser.add_argument("--cellfactor", type=float, default=1, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=500, help="Height of iceberg in meters")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--nt", type=int, default=1000, help="number of timesteps")
parser.add_argument("--Ttop", type=float, default=-5, help="Temperature in Celsius at top")
parser.add_argument("--Tbot", type=float, default=-5, help="Temperature in Celsius at bottom")
parser.add_argument("--nondim_length", type=float, default=5, help="Length of iceberg")
parser.add_argument("--tol", type=float, default=5e-6, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=1, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=800, help="Maximum number of solver iterations")
parser.add_argument("--sealevel", type=float, default=0.9, help="Non dimensional water level for hydrostatic pressure")
parser.add_argument("--Kic", type=float, default=100, help="Kic")
parser.add_argument("--strength0", type=float, default=200, help="Tensile strength at 0C")
parser.add_argument("--strength_deg", type=float, default=20, help="Tensile strength degradation per degree C")
parser.add_argument("--save_bp", type=bool, default=False, help="Save bp files")
parser.add_argument("--relax_time", type=float, default=400, help="Total relaxation time days")
parser.add_argument("--lfactor", type=float, default=2.0, help="Multiply l by in lower part of domain")

args = parser.parse_args()


filename = "iceshelf_L" + str(args.nondim_length) + "_H" + str(args.height) \
                        + "_l" + str(args.lstar) \
                        +"_dt" + str(args.dt) + "_relaxt" + str(args.relax_time) \
                        + "_sigmacdeg" + str(args.strength_deg)+ "_sigmac0" + str(args.strength0) \
                    + "_level" + str(args.level) + "_Kic" + str(args.Kic)\
                    + "_cellfactor" + str(args.cellfactor)\
                            + "_Ttop" + str(abs(args.Ttop)) + "_Tbot" + str(abs(args.Tbot)) \
                            + "_lfactor" + str(args.lfactor) \
                            + "_" + args.suffix + "_"


path = './outputs'
os.makedirs(path, exist_ok=True)



msh = kr.meshes.fenicsx_refined_mesh(args.nondim_length, args.lstar/args.cellfactor, 0.3, large_size=0.2, top_fine_length=2.5, htop2 =1.1)
model = kr.base.Simulation(msh)

model.tol = args.tol
model.min_its = args.min_its
model.max_its = args.max_its

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.T = args.Tbot + (args.Ttop - args.Tbot)*z
model.params.A0.value = mf.rate_factor_np(args.Ttop)
model.params.H.value = args.height
# model.params.l.value = args.lstar*args.height
model.params.dt.value = args.dt*24*60*60
model.params.Kic.value = args.Kic*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = args.level
model.params.sea_level.value = args.sealevel * args.height
model.params.length.value = args.nondim_length * args.height

model.params.σc = args.strength0*1e3 - args.strength_deg*1e3*(model.params.T)

def smoothstep(x, x_c, width):
    return 0.5*(1 + ufl.tanh((x-x_c)/width))

def smoothtransition(a, b, x, x_c, width):
    return a + (b-a)*smoothstep(x, x_c, width)

if args.lfactor>1.0:
    model.params.l = smoothtransition(args.lstar*args.height*args.lfactor, args.lstar*args.height, x[1], 1 - 0.125, 0.05)
else:
    model.params.l.value = args.lstar*args.height

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
    return (x[0]<(args.nondim_length -0.3))*(x[1]<0.9) | (x[0]<(args.nondim_length - 2.0))




d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, end_cracks, 1.0),
                #   bc.internal_bc(V, lambda x: (x[1]>(1-height))*(1-end_cracks(x)), 0.0),
                ]


model.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])




model.damage_on = False


if args.relax_time > 0:
    nt = 10
    model.params.dt.value = args.relax_time*24*60*60 / nt
    for i in range(nt):
        if MPI.COMM_WORLD.rank == 0:
            print("Relaxation iteration: ", i)
        flag,nits = model.fixed_point(save=True)
        if flag == -1:
            break
        model.momentum.timestep()

    model.params.dt.value = args.dt*24*60*60



t = 0.0
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)





model.damage_on = True



def crack(x,x_c,height=0.06):
    width = args.lstar/args.cellfactor*1
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]>(1-height))

end_crack_x_cs = np.linspace(args.nondim_length-2, args.nondim_length-0.1, 20)
height = 0.08
def end_cracks(x):
    val = np.zeros(x.shape[1],dtype=bool)
    for x_c in end_crack_x_cs:
        val += crack(x,x_c,height)
    return val


model.damage.w.sub(0).interpolate(end_cracks)


for i in range(1,args.nt):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

    flag,nits = model.fixed_point(save=False, stop_bottom=True)

    t += model.params.dt.value
    if args.save_bp:
        model.write_checkpoint(path + "/" + filename +".bp", t)

    A = mf.rate_factor(model.params.T)/model.params.A0

    η0 = mf.viscosity(ufl.dev(mf.ε(model.momentum.vel_prev_it)), 3.0, model.params.viscosity_tol, A=A)


    if i ==1 or i % 10 == 0 or flag == -1 or nits > 6:
        kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
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
                                    t=i)
    
    if flag == -1:
        break

    
    model.timestep()
    # model.momentum.timestep()



if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)


   
