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

args = parser.parse_args()


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def top_boundary(x):
    return np.isclose(x[1], nondim_height - Hw)

def all_boundaries(x):
    return left_boundary(x) + right_boundary(x) + bottom_boundary(x) + top_boundary(x)
def crack(x):
    x_c = nondim_length/2 - 0.5*nondim_height
    width = args.l / (L)
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]<-0.5)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.9*nondim_height))# + (x[1]<(0.1-0.9*refineH[1]))


true_length = 16e3
true_height = 300




L = true_height
l = args.l


aspect_ratio_x = int(25/l)



nondim_length = true_length/L
nondim_height = true_height/L

ρi = 900
ρw = 1000

refineH = (2.5,0.2)
msh = kr.utilities.create_refined_mesh(nondim_length,nondim_height, l/L, ρi/ρw,
                                     aspect_ratios=(aspect_ratio_x,1), refine=refineH,
                                     cell_factor=1)

# add slope to mesh
slope = 0
msh.geometry.x[:,1] = msh.geometry.x[:,1]*(1- slope*msh.geometry.x[:,0])
# msh.geometry.x[:,1] += 0.5

no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0),
                  bc.internal_bc(V, crack, 1.0)
                #   bc.get_zero_bc(V.sub(1), all_boundaries)
                  ]


u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

if args.type == "degraded":
    elast = kr.momentum.elastic.ElasticDegraded
else:
    elast = kr.momentum.elastic.Elasticity

model = kr.base.Simulation(msh, 
                           elast,
                           kr.damage.higherorder.HigherOrder,
                            [u_bc, bc_d], level=0.00)

# model = kr.models.elasticity.elastic_damage(msh, [u_bc,no_bc])

model.params.L.value = L
model.params.l.value = l
model.params.dt.value = 60*60*2
model.params.patm.value = 0.0
model.params.ρi.value = ρi
model.params.ρw.value = ρw
model.params.g.value = 9.8


model.params.ψcrit.value = 1.0
model.params.Gc.value = 0.5

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%


model.setup()



model.fixed_point(min_its=3,solve_damage=True,max_its=200,tol=1e-5)
import adios4dolfinx

filename = './scripts/{}elastic_l{}.bp'.format(
    args.type,
    l,
    model.params.Gc.value,
    model.params.ψcrit.value
)
adios4dolfinx.write_mesh(filename, msh)
adios4dolfinx.write_function(filename, model.momentum.u, name="u")
adios4dolfinx.write_function(filename, model.damage.w, name="w")
kr.utilities.write_xdmf("./outputs/elastictest.xdmf",
                            msh, [model.momentum.u, model.damage.d, 
                                  model.free_energy_plus(model.momentum.ε_e, model.params.ν)],
                            ["u", "d", "psi_plus"])
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

