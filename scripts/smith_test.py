#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.parameters import Params_no_uc, Params_with_uc
import kraken.boundaryconditions as bc_bottom
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.mainclass as mc
import kraken.oneclass as oc

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], L) | np.isclose(x[0], 0)

def bottom_boundary(x):
    return np.isclose(x[1], 0)


hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



# material = Material_no_uc()
material = Params_with_uc()
material.l = 1
# material.Gc = 20.0
L = 100
material.uc = material.L
material.ψcrit = 0.066
material.uc = material.L

# material.g =1e-12
# material.g = 1000

g = 9.8
δ = 1 - material.ρi/material.ρw
Rxx = δ*g*material.ρi*L/2


cell_size = material.l/2.1


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([L, L])],
                            [int(L/cell_size),int(L/cell_size)], mesh.CellType.quadrilateral)



v_bc = lambda V: [
    # bc.get_zero_bc(V.sub(0), left_boundary),
                     bc_bottom.get_bc(V.sub(0), right_boundary, 0.0),
                     bc_bottom.get_zero_bc(V.sub(1), bottom_boundary)]
no_bc = lambda V: []

# bc_d = lambda V: [bc.internal_bc(V, lambda x: x<(x_change+0.1), 0.0)]


model = oc.viscoelastic_damage(msh, [v_bc,no_bc,no_bc], material, 
                               dt = 1)#,g = lambda d: mf.degradation_Lo2023(d,0.05))

import ufl
facets = mesh.locate_entities_boundary(model.msh, model.msh.topology.dim - 1, right_boundary)
mesh_tags = mesh.meshtags(model.msh, model.msh.topology.dim - 1, facets, 1)
ds = ufl.Measure("ds", domain=model.msh, subdomain_data=mesh_tags)
model.ds = ds(1)
x = ufl.SpatialCoordinate(model.msh)
model.p = -Rxx + material.ρi*material.g*(L-x[1])
# # model.material.g = 0.1
# model.bounded = False
# model.setup_elastic()


#%%


model.setup_elastic()
model.setup_damage()

model.solve_elastic()
# model.solve_damage()

utilities.write_xdmf(path + "smith.xdmf",msh,\
                    [model.v,model.d,
                      mf.principal_stress(mf.ε(model.v),material.λ,material.μ),
                      mf.free_energy_plus_spectral(mf.ε(model.v),material.λ,material.μ),
                      mf.free_energy_plus_amor(mf.ε(model.v),material.λ,material.μ),
                        mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,1),
                        mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,5),
                        mf.free_energy_plus_stocek(mf.ε(model.v),material.λ,material.μ),
                      mf.cauchy_stress(mf.ε(model.v),material.λ,material.μ),
                      mf.history_function(mf.ε(model.v),model.Hprev,material.λ,material.μ,material.ψcrit) \
                      ],\
                    ["v","d", "λ","spectral","amor","star1","star5","stocek","stress","history"],t=0)


# model.fixed_point_simple(tol=1e-8)
#%%

# tol = 0.001  # Avoid hitting the outside of the domain
# y = np.linspace(-0 + tol, 1 - tol, 101)
# points = np.zeros((3, 101))
# points[1] = y
# import ufl
# from kraken.numerics.invariants import eigenstate2

# λ, E = eigenstate2(ufl.dev(mf.ε(model.v)))

# points_on_proc,func_vals = utilities.extract_line(points,msh,
#     [mf.free_energy_plus_spectral(mf.ε(model.v),material.λ,material.μ),
#     mf.free_energy_plus_amor(mf.ε(model.v),material.λ,material.μ),
#     mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,1),
#     mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,5),
#     mf.free_energy_plus_stocek(mf.ε(model.v),material.λ,material.μ),
#     mf.cauchy_stress(mf.ε(model.v),material.λ,material.μ)[0,0],
#     λ[0],
#     λ[1],
#     model.pw,

#                              ])


# import matplotlib.pyplot as plt
# fig = plt.figure()
# # plt.plot(func_vals[0], points_on_proc[:, 1], "r-", linewidth=2, label="Displacement")
# plt.plot(func_vals[0], points_on_proc[:, 1], "b-", linewidth=1, label="Spectral")
# # plt.plot(func_vals[1], points_on_proc[:, 1], "g-", linewidth=1, label="Amor")
# # plt.plot(func_vals[2], points_on_proc[:, 1], "m-", linewidth=1, label="Star1")
# # plt.plot(func_vals[3], points_on_proc[:, 1], "c-", linewidth=1, label="Star5")
# # plt.plot(func_vals[4], points_on_proc[:, 1], "k-", linewidth=1, label="Stocek")
# plt.figure()
# plt.plot(func_vals[5], points_on_proc[:, 1], "r-", linewidth=1, label="Stress")

# # plt.figure()
# # plt.plot(func_vals[6], points_on_proc[:, 1], "r-", linewidth=1, label="\lambda1")
# # plt.plot(func_vals[7], points_on_proc[:, 1], "g-", linewidth=1, label="\lambda2")


# plt.grid(True)
# plt.ylabel("y")
# plt.legend()
