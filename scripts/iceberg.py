#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
from dolfinx import io, mesh
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=float, default=0.00, help="Water level in cracks, non dimensional")
parser.add_argument("--split", type=str, default="lo", help="Energy split to use")
parser.add_argument("--l", type=float, default=2, help="Regularization length scale in meters")
parser.add_argument("--dt", type=float, default=3, help="Time step in days")
parser.add_argument("--cellfactor", type=float, default=2, help="Mesh cell size factor")
parser.add_argument("--psicrit", type=float, default=1.0, help="Critical energy threshold")
parser.add_argument("--height", type=float, default=300, help="Height of iceberg in meters")
parser.add_argument("--Gc", type=float, default=1.0, help="Gc")
parser.add_argument("--type", type=str, default="relaxation", help="gravity loop initilisation or relaxation first")
parser.add_argument("--damagemodel", type=str, default="AT2higher", help="damage model to use")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--gv_tol", type=float, default=1e-5, help="tolerance for viscous degradation")
parser.add_argument("--nt", type=int, default=50, help="number of timesteps")

args = parser.parse_args()
level = args.level
split = args.split

filename = args.type + "_" + args.split + "_level" + str(level) + "height" + str(args.height) +"Gc" + str(args.Gc)\
                     +"dt" + str(args.dt) + "psicrit" + str(args.psicrit)\
                        + "l" + str(args.l) + "cellfactor" + str(args.cellfactor)\
                            + "gv_tol" + str(-np.log10(args.gv_tol)) + \
                            "_damagemodel" + args.damagemodel + "_" + args.suffix + "_"



if MPI.COMM_WORLD.rank == 0:
    print("Level: ", level)
    print("Split: ", split)
    print("Regularization length scale (m): ", args.l)
    print("Time step (days): ", args.dt)
    print("Mesh cell size factor: ", args.cellfactor)
    print("Critical energy threshold: ", args.psicrit)
    print("Height (m): ", args.height)
    print("Gc: ", args.Gc)
    print("damage model: ", args.damagemodel)

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -flotation_height)

# def crack(x):
#     x_c = nondim_length/2 - nondim_height
#     l = params.lstar
#     return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.98*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))



true_length = 16e3
true_height = args.height

L = true_height
l = args.l
ρi = 900
ρf = 350
ρsw = 1000
D = 32.5


# path = '/data/hpcdata/users/dancha/outputs'
path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = true_length/L
nondim_height = true_height/L

# flotation_height = mf.flotation_height(ρi/ρsw,ρf/ρsw,D/L)
flotation_height = ρi/ρsw

refineH = (2.0,0.2)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L, flotation_height,
                                     aspect_ratios=(100,1), refine=refineH,
                                     cell_factor=args.cellfactor, cell_type=mesh.CellType.triangle)


# msh = kr.utilities.create_iceberg_gmsh_mesh(l/(args.cellfactor*L), [2.5, 0.4, 0.15], true_length/(2*L), ρi/ρsw)


d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0)]


u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1), left_boundary)
                        ]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
if args.damagemodel == "AT1lower":
    damage_model = kr.damage.lowerorder.Bounded
elif args.damagemodel == "AT1higher":
    damage_model = kr.damage.higherorder.PenalizedAT1
elif args.damagemodel == "AT2higher":
    damage_model = kr.damage.higherorder.HigherOrder
elif args.damagemodel == "AT2higher_penalized":
    damage_model = kr.damage.higherorder.PenalizedAT2
elif args.damagemodel == "AT2lower":
    damage_model = kr.damage.lowerorder.NonLinear
elif args.damagemodel == "AT2higher_bounded":
    damage_model = kr.damage.higherorder.Bounded

model = kr.base.Simulation(msh,
                           kr.momentum.mixed.SemiLagrangianEpsilon,
                           damage_model, [u_bc, d_bc], 
                           level=level, split=split)


# model.T = mf.temperature(msh,ρi/ρsw,-30,-2)
model.T = -10.0
model.params.L.value = L
model.params.l.value = args.l
model.params.dt.value = args.dt*24*60*60
model.params.ρi.value = ρi
model.params.ρw.value = ρsw
model.params.ψcrit.value = args.psicrit
model.params.Gc.value = args.Gc
model.params.patm.value = 0.0
model.params.gv_tol.value = args.gv_tol


#%%
model.setup()


if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)

from dolfinx import fem
import ufl
min_its = 3
tol = 1e-5
if args.type == "relaxation":
    solve_d = False
else:
    solve_d = True



for i in range(args.nt):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)


    if i<10 and args.type == "relaxation":
        model.params.dt.value = 10*24*60*60
    else:
        model.params.dt.value = args.dt*24*60*60

    if i == 10 and args.type == "relaxation":
        solve_d = True

    

    errors = model.fixed_point(min_its=min_its, tol=tol, max_its=200, solve_damage=solve_d)
    
    model.write_checkpoint(path + "/" + filename +".bp", t=i)

    kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                      model.momentum.u_e, model.momentum.u_v,
                                    ],
                                    ["u","d",
                                    "ue","uv"
                                    ],
                                  t=i)


    if solve_d:
        model.damage.timestep()
    model.momentum.timestep()
    
    

   
