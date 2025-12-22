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
parser.add_argument('--notch', type=int, default=1, help='1 for notch, 0 for no notch')

Lx = 1e3
H = 300
h = 0.025

nondim_length = Lx / H
nondim_height = 1.0
flotation_height = 0.9

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length)

def bottom_boundary(x):
    return np.isclose(x[1], -flotation_height)

# def crack(x):
#     x_c = nondim_length/2 - args.crack_x*nondim_height
#     width = args.l/args.cellfactor / args.height
#     return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]<-0.85)

def crack(x):
    x_c = nondim_length/2
    hz = 0.3
    crack_width = h
    return (x[0]>(x_c-crack_width))*(x[0]<(x_c+crack_width))*(x[1]>1-flotation_height-hz)


path = './outputs'
os.makedirs(path, exist_ok=True)


notched = bool(parser.parse_args().notch)

meshpath = "./meshes"

if notched:
    filename = "notched.xdmf"
    msh = io.XDMFFile(MPI.COMM_WORLD, meshpath + "/" + filename, "r").read_mesh()
else:
    filename = "no_notch.xdmf"
    msh = io.XDMFFile(MPI.COMM_WORLD, meshpath + "/" + filename, "r").read_mesh()

#load msh from xdmf



if notched:
    d_bc = lambda V: []
else:
    d_bc = lambda V: [bc.internal_bc(V, crack, 1.0)]




u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1), left_boundary),
                        # bc.get_zero_bc(V.sub(0).sub(1), bottom_boundary),
                        # bc.get_zero_bc(V.sub(1).sub(1), bottom_boundary)
                        ]


model = kr.base.Simulation(msh,
                           kr.momentum.mixed.SemiLagrangianEpsilon,
                          kr.damage.higherorder.HigherOrder, [u_bc, d_bc], 
                           )


model.params.T.value = -10
# model.params.T.value = args.T
model.params.A0.value = mf.rate_factor_np(model.params.T.value)
model.params.L.value = H
model.params.l.value = h*2
model.params.dt.value = 0.05*24*60*60
model.params.ρi.value = 900
model.params.ρw.value = 1000
model.params.ψcrit.value = 1.0
model.params.Gc.value = 1.0
model.params.patm.value = 0.0
model.params.gv_tol.value = 1e-4


#%%




model.setup()
# model.damage.w.sub(0).interpolate(crack)
model.damage.solve()

        # model.fixed_point(min_its=3, tol=1e-6, max_its=100, solve_damage=False)
        # model.damage.timestep()

if notched:
    filename = "notched"
else:
    filename = "no_notch"



t = 0.0
model.write_checkpoint(path + "/" + filename +".bp", t)






for i in range(1,50):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)


    
    
    # flag = model.fixed_point(min_its=3, tol=1e-6, max_its=100, solve_damage=False, save=False)
    model.momentum.solve()
    

    t += model.params.dt.value
    model.write_checkpoint(path + "/" + filename +".bp", t)

    λ = mf.largest_eigenvalue(es.cauchy_stress(model.momentum.ε_e, model.params.ν))

    kr.utilities.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                    model.momentum.u_e, model.momentum.u_v,
                                    λ,
                                    model.momentum.ψplus
                            ],
                                    ["u","d",
                                    # "ε_e","ε_v","ε",
                                    "ue","uv",
                                    "λ",
                                    "ψplus"
                                    ],
                                t=i)
  
    model.momentum.timestep()

    

   
