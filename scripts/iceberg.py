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
parser.add_argument("--lstar", type=float, default=0.06, help="Regularization length scale in meters")
parser.add_argument("--dt", type=float, default=3, help="Time step in days")
parser.add_argument("--cellfactor", type=float, default=1, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=300, help="Height of iceberg in meters")
parser.add_argument("--type", type=str, default="relaxation", help="gravity loop initilisation or relaxation first")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--nt", type=int, default=200, help="number of timesteps")
parser.add_argument("--Ttop", type=float, default=-20, help="Temperature in Celsius at top")
parser.add_argument("--Tbot", type=float, default=-20, help="Temperature in Celsius at bottom")
parser.add_argument("--nondim_length", type=float, default=20, help="Length of iceberg")
parser.add_argument("--tol", type=float, default=5e-6, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=1, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=400, help="Maximum number of solver iterations")
parser.add_argument("--sealevel", type=float, default=0.9, help="Non dimensional water level for hydrostatic pressure")
parser.add_argument("--Kic", type=float, default=100, help="Kic")
parser.add_argument("--strength0", type=float, default=200, help="Tensile strength at 0C")
parser.add_argument("--strength_deg", type=float, default=20, help="Tensile strength degradation per degree C")
parser.add_argument("--no-cracks", type=bool, default=False, help="Don't include initial cracks")
parser.add_argument("--save_bp", type=bool, default=False, help="Save bp files")

args = parser.parse_args()


filename = args.type + "_level" + str(args.level) + "_height" + str(args.height) +"_Kic" + str(args.Kic)\
                     +"_dt" + str(args.dt) + "_sigmacdeg" + str(args.strength_deg)\
                        + "_sigmac0" + str(args.strength0) \
                        + "_lstar" + str(args.lstar) + "_cellfactor" + str(args.cellfactor)\
                            + "_Ttop" + str(abs(args.Ttop)) + "_Tbot" + str(abs(args.Tbot)) + \
                            "_nondimlength" + str(args.nondim_length) + "_" + args.suffix + "_"



if MPI.COMM_WORLD.rank == 0:
    # print dolfinx version
    print("Dolfinx version: ", dolfinx.__version__)
    print("Level: ", args.level)
    print("Regularization length scale star: ", args.lstar)
    print("Time step (days): ", args.dt)
    print("Mesh cell size factor: ", args.cellfactor)
    print("Height (m): ", args.height)

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length)

def bottom_boundary(x):
    return np.isclose(x[1], 0)#*(x[0]<=(nondim_length - nondim_height))

def bottom_left(x):
    return np.isclose(x[1], 0)*(x[0]<=(nondim_length/2))

def bottom_right(x):
    return np.isclose(x[1], 0)*(x[0]>=(nondim_length/2))

def top_boundary(x):
    return np.isclose(x[1], 1.0)

def single_dof(x):
    return np.isclose(x[0], nondim_length/2)*np.isclose(x[1], 0.5)

def crack(x,x_c,height=0.06):
    width = args.lstar/args.cellfactor*1
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]>(1-height))

def basal_crack(x,x_c,height=0.5):
    width = args.lstar/args.cellfactor*1
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]<height)

def fixed(x):
    return (x[0]<(nondim_length - args.refine_x*0.98*nondim_height))# + (x[0]>(nondim_length - nondim_height/2))



path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = args.nondim_length
nondim_height = 1.0



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



model = kr.base.Simulation(msh,split="lo_p")

model.tol = args.tol
model.min_its = args.min_its
model.max_its = args.max_its

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.Ttop.value = args.Ttop
model.params.Tbot.value = args.Tbot
model.params.A0.value = mf.rate_factor_np(args.Ttop)
model.params.H.value = args.height
model.params.l.value = args.lstar*args.height
model.params.dt.value = args.dt*24*60*60
model.params.Kic.value = args.Kic*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = args.level
model.params.sea_level.value = args.sealevel * args.height
model.params.length.value = args.nondim_length * args.height
# model.params.ge_tol.value = 1e-3

model.params.σt_deg.value = args.strength_deg*1e3
model.params.σt0.value = args.strength0*1e3



# model.params.set_Gc_from_Kic()
# model.params.set_psicrit_from_σc()


if MPI.COMM_WORLD.rank == 0:
    print(model.params.ucstar_float )



# if args.type == "ssa":
#     d_bc = lambda V: [bc.internal_bc(V, lambda x: (x[0]<=0.09) + (x[0]>=nondim_length-0.09), 0.0)]



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
    δ = model.params.δ; ν = model.params.ν
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
elif args.type == "icebergsymm" or args.type == "relaxation":
      u_bc = lambda V: [
                            # bc.internal_point(V.sub(0).sub(0), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
                    #   bc.internal_point(V.sub(1).sub(0), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
                    #   bc.internal_point(V.sub(1).sub(1), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
                            bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                            bc.get_zero_bc(V.sub(1).sub(0), left_boundary),

                            ]



else:
    u_bc = lambda V: [
                            bc.internal_point(V.sub(0).sub(0), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
                      bc.internal_point(V.sub(1).sub(0), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
                      bc.internal_point(V.sub(1).sub(1), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
                            # bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                            # bc.get_zero_bc(V.sub(1).sub(0), left_boundary),

                            ]





basal_crack_spacing = 0.8
n_cracks = int((nondim_length-basal_crack_spacing)//basal_crack_spacing)
crack_x_cs = np.linspace(basal_crack_spacing/2, nondim_length-basal_crack_spacing/2, n_cracks*4 -3)
# crack_x_cs += cell_size/2
def surface_cracks(x):
    val = np.zeros(x.shape[1],dtype=bool)
    for x_c in crack_x_cs:
        val += crack(x,x_c)
    return val



basal_crack_x_cs = np.linspace(basal_crack_spacing/2,nondim_length-basal_crack_spacing/2, n_cracks)
# basal_crack_x_cs += cell_size/2
def basal_cracks(x):
    val = np.zeros(x.shape[1],dtype=bool)
    for x_c in basal_crack_x_cs:
        val += basal_crack(x,x_c,height=0.4)
    return val

cracks = lambda x: surface_cracks(x) + basal_cracks(x)

if args.no_cracks or args.type == "ssa":
    d_bc = lambda V: []
else:
    d_bc = lambda V: [bc.internal_bc(V, cracks, 1.0)]



model.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])


if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)

model.damage_on = False
if args.type == ("relaxation" or "chop"):
    i_start = 10
    model.params.dt.value = 25*24*60*60
else:
    i_start = 1
    


t = 0.0
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)

if args.type == "ssa":
    model.momentum.solve()
    model.damage.w.sub(0).interpolate(cracks)
    
#     # model.momentum.solve()
#     model.damage_on = True
#     model.fixed_point(save=True)
#     model.damage_on = False
#     model.damage.timestep()

#     u_bc = lambda V: [bc.internal_point(V.sub(0).sub(0), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),
#                       bc.internal_point(V.sub(1).sub(0), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),m
#                       bc.internal_point(V.sub(1).sub(1), lambda x: left_boundary(x)*bottom_boundary(x), 0.0),]
#     model.momentum.update_bcs(u_bc)model.momentum.solve()


# model.momentum.solve()
# model.damage.solve()



for i in range(1,args.nt):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)

   
    if i == i_start:

        if args.type == "chop":
        
                cells_subdomain = mesh.locate_entities(model.msh, model.msh.topology.dim, lambda x: x[0]<basal_crack_x_cs[-6])

                submesh,parent_cells,_,_ = mesh.create_submesh(model.msh, model.msh.topology.dim, cells_subdomain)

                submodel = kr.base.Simulation(submesh,split="lo_p")
                
                submodel.interpolate_from_parent(model,parent_cells, [u_bc, d_bc])

                model = submodel
        # if MPI.COMM_WORLD.rank == 0:
        #     print(model.params.ucstar_float )

        # model.momentum.solve()
        # model.damage.w.sub(0).interpolate(cracks)
        model.params.dt.value = args.dt*24*60*60
        model.damage_on = True

        

    flag = model.fixed_point(save=True)

    t += model.params.dt.value
    if args.save_bp:
        model.write_checkpoint(path + "/" + filename +".bp", t)
    ψpold = es.free_energy_plus_lo(model.momentum.ε_e, model.params.ν)
    kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                            model.msh, [model.momentum.du,model.damage.d,model.damage.d_prev_it2,model.damage.d_prev_it,model.damage.d_prev_it3,
                                    model.momentum.u_v, model.momentum.u_e,
                                    model.momentum.ψplus/model.params.ψcritstar,
                                    ψpold/model.params.ψcritstar,
                                    model.momentum.ε_e,
                                    model.params.Gc,
                                    model.params.T,
                                    model.params.ψcrit,
                                    ],
                                    ["u","d","dprev2","dprev","dprev3",
                                    "uv","ue",
                                    "psi_plus",
                                    "psi_plus_old",
                                    "eps_e",
                                    "Gc",
                                    "T",
                                    "ψcrit",
                                    ],
                                t=i)
    
    if flag == -1:
        break

    
    model.timestep()
    

    

   
