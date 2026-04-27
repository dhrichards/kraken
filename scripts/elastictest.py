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
parser.add_argument("--type", type=str, default="normal", help="degraded or normal")
parser.add_argument("--cellfactor", type=float, default=1.0, help="Mesh cell size factor")

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
true_height = 100




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
                        [nondim_length/2, nondim_height]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)

# msh = kr.utilities.create_iceberg_gmsh_mesh(
#     args.l/(args.cellfactor*L), 
#     [2, 0.25, 0.125], 
#     nondim_length/2, 
#     ρi/ρw
#     )

# add slope to mesh
slope = 0
msh.geometry.x[:,1] = msh.geometry.x[:,1]*(1- slope*msh.geometry.x[:,0])
# msh.geometry.x[:,1] += 0.5

no_bc = lambda V: []
bc_d = lambda V: [
    # bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, crack, 1.0)
                #   bc.get_zero_bc(V.sub(1), all_boundaries)
                  ]

if args.type == "pressure":
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary)]
else:
    u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                    # bc.get_zero_bc(V.sub(1), bottom_boundary)
                    ]
               

if args.type == "degraded":
    elast = kr.momentum.elastic.ElasticDegraded
elif args.type == "normal":
    elast = kr.momentum.elastic.Elasticity
elif args.type == "pressure":
    elast = kr.momentum.elastic.ElasticPressure
elif args.type == "3D":
    elast = kr.momentum.elastic.Elastic3D
elif args.type == "energy":
    elast = kr.momentum.elastic.ElasticEnergySplit

model = kr.base.Simulation(msh, 
                           elast,
                           kr.damage.higherorder.AT2,
                            [u_bc, bc_d])

model.params.H.value = L
model.params.l.value = l
model.params.dt.value = 60*60*2
model.params.patm.value = 0
model.params.Gc.value = 1.0
model.params.crack_level_above_sea.value = 0.0
model.params.sea_level.value = 0.9 * true_height

model.params.ψcrit.value = 1.0



# B = -0.34
B = -0.45
model.params.friction_angle.value = np.arcsin(3*np.sqrt(3)*B/(2-np.sqrt(3)*B))

#%%

model.damage_on = True
model.setup()

model.momentum.solve()


# model.msh.geometry.x[:,:model.msh.geometry.dim] -= model.params.ucstar_float*model.momentum.u.x.array.reshape((-1, model.msh.geometry.dim))
        



# model.fixed_point(save=True)
import adios4dolfinx

# filename = './scripts/{}elastic_l{}.bp'.format(
#     args.type,
#     l,
#     model.params.Gc.value,
#     model.params.ψcrit.value
# )
# adios4dolfinx.write_mesh(filename, msh)
# adios4dolfinx.write_function(filename, model.momentum.u, name="u")
# adios4dolfinx.write_function(filename, model.damage.w, name="w")


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


    
#%%
# from matplotlib import tri
# from dolfinx import fem


# #gather data and save to npz

# CG1 = fem.functionspace(msh, ("CG", 1))
# ux = fem.Function(CG1)
# uz = fem.Function(CG1)
# d = fem.Function(CG1)

# ux.interpolate(fem.Expression(model.momentum.u.sub(0), 
#                              CG1.element.interpolation_points()))

# uz.interpolate(fem.Expression(model.momentum.u.sub(1), 
#                              CG1.element.interpolation_points()))

# d.interpolate(fem.Expression(model.damage.d, 
#                              CG1.element.interpolation_points()))


# connty = msh.topology.connectivity(2, 0)
# connty_array = np.array([connty.links(i) 
#         for i in range(connty.num_nodes)])
# tess = tri.Triangulation(
#         msh.geometry.x[:,0], 
#         msh.geometry.x[:,1], 
#         triangles=connty_array)

# x = msh.geometry.x[:,0]
# z = msh.geometry.x[:,1]

# filename = 'elastic_l{}_Gc{}_psicrit{}.npz'.format(
#     l,
#     model.params.Gc.value,
#     model.params.ψcrit.value
# )

# np.savez(filename, x=x, z=z, contty=connty_array,
#          ux=ux.x.array, uz=uz.x.array, d=d.x.array)

