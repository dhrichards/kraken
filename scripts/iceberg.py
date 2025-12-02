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
parser.add_argument("--cellfactor", type=float, default=1, help="Mesh cell size factor")
parser.add_argument("--psicrit", type=float, default=1.0, help="Critical energy threshold")
parser.add_argument("--height", type=float, default=300, help="Height of iceberg in meters")
parser.add_argument("--Gc", type=float, default=0.5, help="Gc")
parser.add_argument("--type", type=str, default="relaxation", help="gravity loop initilisation or relaxation first")
parser.add_argument("--damagemodel", type=str, default="AT2higher", help="damage model to use")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--gv_tol", type=float, default=1e-3, help="tolerance for viscous degradation")
parser.add_argument("--nt", type=int, default=200, help="number of timesteps")
parser.add_argument("--crack_x", type=float, default=0.5, help="x position of crack center from end (non dimensional)")
parser.add_argument("--rhoi", type=float, default=900, help="ice density")
parser.add_argument("--rhow", type=float, default=1000, help="water density")
parser.add_argument("--arz", type=float, default=1, help="aspect ratio in z direction")
parser.add_argument("--T", type=float, default=-10, help="Temperature in Celsius")
parser.add_argument("--refine_x", type=float, default=2.0, help="refinement in x direction near crack")
parser.add_argument("--refine_z", type=float, default=0.2, help="refinement in z direction near crack")
parser.add_argument("--length", type=float, default=16000, help="Length of iceberg in meters")
parser.add_argument("--tol", type=float, default=1e-6, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=3, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=500, help="Maximum number of solver iterations")


args = parser.parse_args()


filename = args.type + "_" + args.split + "_level" + str(args.level) + "height" + str(args.height) +"Gc" + str(args.Gc)\
                     +"dt" + str(args.dt) + "psicrit" + str(args.psicrit)\
                        + "l" + str(args.l) + "cellfactor" + str(args.cellfactor)\
                            + "gv_tol" + str(-np.log10(args.gv_tol)) + \
                            "_damagemodel" + args.damagemodel + "_" + args.suffix + "_"



if MPI.COMM_WORLD.rank == 0:
    print("Level: ", args.level)
    print("Split: ", args.split)
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
#     x_c = nondim_length/2 - args.crack_x*nondim_height
#     width = args.l/args.cellfactor / args.height
#     return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]<-0.85)

def crack(x,x_c):
    width = args.l/args.cellfactor / args.height
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]>0.07)

def fixed(x):
    return (x[0]<(nondim_length/2 - args.refine_x*0.98*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))



path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = args.length/args.height
nondim_height = 1.0

flotation_height = args.rhoi/args.rhow


aspect_ratio_x = int(300/args.l)

msh = kr.utilities.create_refined_mesh(args.length/args.height, 1.0, args.l/args.height, flotation_height,
                                     aspect_ratios=(aspect_ratio_x,args.arz), refine=(args.refine_x,args.refine_z),
                                     cell_factor=args.cellfactor, cell_type=mesh.CellType.triangle)


# msh = kr.utilities.create_iceberg_gmsh_mesh(l/(args.cellfactor*L), [2.5, 0.4, 0.15], true_length/(2*L), ρi/ρsw)




d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, cracks, 1.0)
                  ]



u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1).sub(0), left_boundary)
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
                           level=args.level, split=args.split)


# model.T = mf.temperature(msh,ρi/ρsw,-30,-2)
model.T = args.T
model.params.L.value = args.height
model.params.l.value = args.l
model.params.dt.value = args.dt*24*60*60
model.params.ρi.value = args.rhoi
model.params.ρw.value = args.rhow
model.params.ψcrit.value = args.psicrit
model.params.Gc.value = args.Gc
model.params.patm.value = 0.0
model.params.gv_tol.value = args.gv_tol


#%%
model.setup()

# crack_spacing = 0.1
# crack_start = 0.2
# crack_end = 1.8
# crack_x_cs = nondim_length/2 - np.arange(crack_start, crack_end, crack_spacing)
# def cracks(x):
#     val = np.zeros(x.shape[1],dtype=bool)
#     for x_c in crack_x_cs:
#         val += crack(x,x_c)
#     return val
# model.damage.w.sub(0).interpolate(lambda x: cracks(x).astype(np.float64))




if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)


if args.type == "relaxation":
    solve_d = False
    model.params.dt.value = 10*24*60*60
else:
    solve_d = True



t = 0.0
model.write_checkpoint(path + "/" + filename +".bp", t)

factors = [40,20,10,5,2.5,1.0]

for i in range(1,args.nt):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)


    if i == 11 and args.type == "relaxation":
        solve_d = True
        model.params.dt.value = args.dt*24*60*60
        
        for factor in factors:
            model.params.Gc.value = args.Gc * factor
            if MPI.COMM_WORLD.rank == 0:
                print("Setting Gc to ", model.params.Gc.value)
            model.fixed_point(min_its=3, tol=args.tol, max_its=50, solve_damage=solve_d)
        
    # if i == 11:
    #     for factor in Gc_factors2:
    #         model.params.Gc.value = args.Gc * factor
    #         if MPI.COMM_WORLD.rank == 0:
    #             print("Setting Gc to ", model.params.Gc.value)
    
    
    flag = model.fixed_point(min_its=args.min_its, tol=args.tol, max_its=args.max_its, solve_damage=solve_d)
    
    # while flag == -1:
    #     model.params.Gc.value *= 2
    #     if MPI.COMM_WORLD.rank == 0:
    #         print("Reverting and setting Gc to ", model.params.Gc.value)
    #     model.revert()
    #     flag = model.fixed_point(min_its=min_its, tol=tol, max_its=200, solve_damage=solve_d)
    
    # while model.params.Gc.value > args.Gc:
    #     model.params.Gc.value /= 2
    #     if MPI.COMM_WORLD.rank == 0:
    #         print("Reducing Gc to ", model.params.Gc.value)
    #     flag = model.fixed_point(min_its=min_its, tol=tol, max_its=500, solve_damage=solve_d)
        


    t += model.params.dt.value
    model.write_checkpoint(path + "/" + filename +".bp", t)

    kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                    model.momentum.u_e, model.momentum.u_v,
                                    model.free_energy_plus(model.momentum.ε_e, model.params.ν),
                                    ],
                                    ["u","d",
                                    "ue","uv",
                                    "psi_plus",
                                    ],
                                t=i)

    if solve_d:
        model.damage.timestep()
    model.momentum.timestep()
    # model.params.dt.value *= 1.05

    

   
