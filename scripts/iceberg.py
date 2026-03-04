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
parser.add_argument("--level", type=float, default=0.00, help="Water level in cracks above sea level (m)")
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
parser.add_argument("--nt", type=int, default=200, help="number of timesteps")
parser.add_argument("--crack_x", type=float, default=0.5, help="x position of crack center from end (non dimensional)")
parser.add_argument("--rhoi", type=float, default=900, help="ice density")
parser.add_argument("--rhow", type=float, default=1000, help="water density")
parser.add_argument("--aspect_ratio_z", type=float, default=1, help="aspect ratio in z direction")
parser.add_argument("--T", type=float, default=-10, help="Temperature in Celsius")
parser.add_argument("--refine_x", type=float, default=2.0, help="refinement in x direction near crack")
parser.add_argument("--refine_z", type=float, default=0.2, help="refinement in z direction near crack")
parser.add_argument("--length", type=float, default=16000, help="Length of iceberg in meters")
parser.add_argument("--tol", type=float, default=5e-6, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=1, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=400, help="Maximum number of solver iterations")
parser.add_argument("--meshtype", type=str, default="refined", help="Type of mesh to use: gmsh or refined")
parser.add_argument("--sealevel", type=float, default=0.9, help="Non dimensional water level for hydrostatic pressure")

args = parser.parse_args()


filename = args.type + "_" + args.split + "_level" + str(args.level) + "height" + str(args.height) +"Gc" + str(args.Gc)\
                     +"dt" + str(args.dt) + "psicrit" + str(args.psicrit)\
                        + "l" + str(args.l) + "cellfactor" + str(args.cellfactor)\
                            + "T" + str(abs(args.T)) +  \
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
    return np.isclose(x[1], 0)#*(x[0]<=(nondim_length/2 - nondim_height))

def bottom_left(x):
    return np.isclose(x[1], 0)*(x[0]<=(nondim_length/4))

def bottom_right(x):
    return np.isclose(x[1], 0)*(x[0]>=(nondim_length/4))

def crack(x,x_c):
    width = args.l/args.cellfactor / args.height
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]>0.97)

def basal_crack(x,x_c,height=0.5):
    width = args.l/args.cellfactor / args.height*0.9
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]<height)

def fixed(x):
    return (x[0]<(nondim_length/2 - args.refine_x*0.98*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))



path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = args.length/args.height
nondim_height = 1.0

aspect_ratio_x = int(300/args.l)

if args.meshtype == "refined":
    msh = kr.meshes.create_refined_mesh(
        args.length/args.height, 1.0, 
        args.l/args.height,
        aspect_ratios=(aspect_ratio_x,args.aspect_ratio_z), 
        refine=(args.refine_x,args.refine_z),
        cell_factor=args.cellfactor, cell_type=mesh.CellType.triangle)


elif args.meshtype == "gmsh":
    msh = kr.meshes.create_iceberg_gmsh_mesh(
    args.l/(args.cellfactor*args.height), 
    [args.refine_x, 0.25, args.refine_z], 
    args.length/(2*args.height), 
    )

elif args.meshtype == "refined2":
    small_size = args.l/(args.cellfactor*args.height)
    L = args.length/2/args.height
    meshfilename = f"refined_mesh_length{L:.2f}_size{small_size:.2f}.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, meshfilename, "r") as xdmf:
        msh = xdmf.read_mesh()


elif args.meshtype == "uniform":
    cell_size = args.l/(args.cellfactor*args.height)
    nx = int((nondim_length/2)/cell_size)
    nz = int(nondim_height/cell_size)
    msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [[0.0, 0.0],
                            [nondim_length/2, nondim_height]],
                            [nx, nz],
                            cell_type=mesh.CellType.triangle)

# bottom_facets = mesh.locate_entities_boundary(msh, msh.topology.dim-1, bottom_boundary)
# right_facets = mesh.locate_entities_boundary(msh, msh.topology.dim-1, right_boundary)
# facets = np.hstack([bottom_facets, right_facets])
# values = np.hstack([np.full_like(bottom_facets, 1), np.full_like(right_facets, 2)])
# sorted_facets = np.argsort(facets)
# mt = mesh.meshtags(msh, msh.topology.dim-1, facets[sorted_facets], values[sorted_facets])
# ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)




if args.meshtype == "uniform":
    d_bc = lambda V: []
else:   
    d_bc = lambda V: [
                bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, cracks, 1.0)
                  ]
    





if args.type == "cliff_frozen":
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(0), bottom_boundary),
                        bc.get_zero_bc(V.sub(1), bottom_boundary),
            ]
elif args.type == "cliff_sliding":
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(0).sub(1), bottom_boundary),
                        bc.get_zero_bc(V.sub(1).sub(1), bottom_boundary),
                        ]
elif args.type == "cliff_stickslip":
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(0), bottom_left),
                        bc.get_zero_bc(V.sub(1), bottom_left),
                        bc.get_zero_bc(V.sub(0).sub(1), bottom_right),
                        bc.get_zero_bc(V.sub(1).sub(1), bottom_right),
                        ]
elif args.type == "ssa":
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(0).sub(1), bottom_boundary),
                        bc.get_zero_bc(V.sub(1).sub(1), bottom_boundary),
                        bc.get_bc(V.sub(1).sub(0), right_boundary, 0.1),
                        ]
else:
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                            bc.get_zero_bc(V.sub(1).sub(0), left_boundary),

                            ]

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
                           split=args.split)

model.tol = args.tol
model.min_its = args.min_its
model.max_its = args.max_its

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
# model.params.T = -1 + (args.T - -1)*z
model.params.T.value = args.T
model.params.A0.value = mf.rate_factor_np(args.T)
model.params.H.value = args.height
model.params.l.value = args.l
model.params.dt.value = args.dt*24*60*60
model.params.ρi.value = args.rhoi
model.params.ρw.value = args.rhow
model.params.ψcrit.value = args.psicrit
model.params.patm.value = 0.0
model.params.Gc.value = args.Gc
model.params.crack_level_above_sea.value = args.level
model.params.sea_level.value = args.sealevel * args.height
model.params.length.value = args.length

model.params.ge_tol.value = 1e-6


# model.params.Gc = (args.Gc - 0.1)*z**2 + 0.1

if MPI.COMM_WORLD.rank == 0:
    print(model.params.ucstar_float )



model.setup()

crack_spacing = 0.1
crack_start = 0.2
crack_end = args.refine_x - 0.05
crack_x_cs = nondim_length/2 - np.arange(crack_start, crack_end, crack_spacing)
def cracks(x):
    val = np.zeros(x.shape[1],dtype=bool)
    for x_c in crack_x_cs:
        val += crack(x,x_c)
    return val


basal_crack_spacking = 0.2
basal_crack_start = 0.2
basal_crack_end = args.refine_x - 0.05
# basal_crack_x_cs = nondim_length/2 - np.arange(basal_crack_start, basal_crack_end, basal_crack_spacking)
basal_crack_x_cs = np.arange(0, nondim_length/2, basal_crack_spacking)
def basal_cracks(x):
    val = np.zeros(x.shape[1],dtype=bool)
    for x_c in basal_crack_x_cs:
        val += basal_crack(x,x_c,height=0.3)
    return val
    




if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)

model.damage_on = False
if args.type == "relaxation":
    i_start = 10
    # model.params.dt.value = 10*24*60*60
else:
    i_start = 1
    



t = 0.0
# model.write_checkpoint(path + "/" + filename +".bp", t)




for i in range(1,args.nt):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)



    if i == i_start:
        
        model.damage_on = True
        # model.damage.w.sub(0).interpolate(lambda x: basal_cracks(x).astype(np.float64))
        # model.momentum.solve()
        # model.damage.timestep()

    
    flag = model.fixed_point(save=True)

    ε = model.momentum.ε_e
    p = model.momentum.water_pressure(model.momentum.u)
    ψplus = es.free_energy_plus_lo(ε, model.params.ν)/model.params.ψcritstar
    

    t += model.params.dt.value
    # model.write_checkpoint(path + "/" + filename +".bp", t)
    g = es.degradation_default(model.damage.d)
    kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                    model.momentum.u_v, model.momentum.u_e,
                                    model.momentum.ψplus/model.params.ψcritstar,
                                    ψplus,
                                    p,
                                    # model.mass.ρ,
                                    ],
                                    ["u","d",
                                    "uv","ue",
                                    "psi_plus",
                                    "psi_plus_mod",
                                    "pressure",
                                    ],
                                t=i)
    
    if flag == -1:
        break

    
    model.timestep()
    

    

   
