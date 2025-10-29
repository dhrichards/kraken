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

args = parser.parse_args()
level = args.level
split = args.split

filename = args.type + "_" + args.split + "_level" + str(level) + "height" + str(args.height) +"Gc" + str(args.Gc)\
                     +"dt" + str(args.dt) + "psicrit" + str(args.psicrit)\
                        + "l" + str(args.l) + "cellfactor" + str(args.cellfactor)+"_damagemodel" + args.damagemodel + "_"



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

# def bottom_boundary(x):
#     return np.isclose(x[1], -Hw)

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

refineH = (2.5,0.4)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L, flotation_height,
                                     aspect_ratios=(300,1), refine=refineH,
                                     cell_factor=args.cellfactor, cell_type=mesh.CellType.triangle)


d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)
                        ]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
if args.damagemodel == "AT1lower":
    damage_model = kr.damage.lowerorder.Bounded
elif args.damagemodel == "AT1higher":
    damage_model = kr.damage.higherorder.Bounded
elif args.damagemodel == "AT2higher":
    damage_model = kr.damage.higherorder.HigherOrder
elif args.damagemodel == "AT2lower":
    damage_model = kr.damage.lowerorder.NonLinear

model = kr.base.Simulation(msh, [u_bc, d_bc],
                           kr.momentum.mixed.SemiLagrangianEpsilon,
                           damage_model, level=level, split=split)


# model.T = mf.temperature(msh,ρi/ρsw,-30,-2)
model.params.L.value = L
model.params.l.value = args.l
model.params.dt.value = args.dt*24*60*60
model.params.ρi.value = ρi
model.params.ρw.value = ρsw
model.params.ψcrit.value = args.psicrit
model.params.Gc.value = args.Gc
model.params.patm.value = 0.0


#%%
min_its = 10

# model.setup_all()
model.setup()


if args.type == "iceberg":
    gs = [5,5.5,6,7,7.5,8,8.5,9,9.4]

    for i,g in enumerate(gs):

        model.params.g.value = g

        model.fixed_point(min_its=min_its, tol=1e-5,max_its=200)

        kr.utilities.write_xdmf(path + "/" + filename + "gravity" + str(i) + ".xdmf",
                                msh, [model.momentum.u,model.damage.d,
                                    #   model.momentum.u_e, model.momentum.u_v,
                                    ],
                                    ["u","d",
                                    "ue","uv"
                                    ],
                                    t=i)
        model.damage.timestep()
 
        model.momentum.timestep()



model.params.g.value = 9.8

if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)

from dolfinx import fem
import ufl
min_its = 5

if args.type == "relaxation":
    solve_d = False
    Gc_loop = args.Gc*np.array([3,2.5,2,1.5,1.25])
else:
    solve_d = True

for i in range(600):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)


    if i<10:
        model.params.dt = 10*24*60*60
    else:
        model.params.dt = args.dt*24*60*60

    
    
    if i == 10 and args.type == "relaxation":
        # tol = 1e-6
        solve_d = True

        # for val in Gc_loop:
        #     if MPI.COMM_WORLD.rank == 0:
        #         print("Setting Gc to ", val)
        #     model.params.Gc.value = val
        #     model.fixed_point(min_its=10, tol=tol, max_its=300, solve_damage=solve_d)
        # if MPI.COMM_WORLD.rank == 0:
        #     print("Setting Gc to ", args.Gc)
        # model.params.Gc.value = args.Gc
    else:
        tol = 1e-5

    if i == 20:
        min_its = 3
    model.fixed_point(min_its=min_its, tol=tol, max_its=200, solve_damage=solve_d)



    kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                            msh, [model.momentum.u, model.damage.d,model.momentum.ρ,
                                  model.momentum.u_e, model.momentum.u_v,
                                #   ufl.div(model.momentum.vel),ufl.div(model.momentum.du_e),
                                  ],
                                  ["u", "d","ρ",
                                "ue", "uv",
                                # "div_uv","div_ue"
                                ],
                                  t=i)

    model.timestep()
   
