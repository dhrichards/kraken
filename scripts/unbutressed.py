#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.parameters import Params_no_uc, Params_with_uc
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken.utilities as utilities
import kraken.mainclass as mc
import kraken.oneclass as oc

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], 0)



true_length = 8e3
true_height = 300




material = Params_no_uc()
# material = Material_with_uc()
material.L = true_height
material.l = 50
material.patm=0.0
# material.A = 1.0
# material.n = 1.0



# material.set_C1_to_one()
# material.set_C1C2_to_one()
material.set_C2_to_one()


# material.L = true_height
# material.τ = 3600*24
nondim_length = true_length/material.L
nondim_height = true_height/material.L

cell_size = material.lstar/2.1



msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([nondim_length/2, nondim_height])],
                            [60,20], mesh.CellType.quadrilateral)



bc_bottom = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                     bc.get_zero_bc(V.sub(1), bottom_boundary)
                     ]
no_bc = lambda V: []


model = oc.viscoelastic_damage(msh, [bc_bottom,bc_bottom,no_bc], material, 
                               dt = 1)#g = lambda d: mf.degradation_Lo2023(d,0.05))


model.p_ext = lambda u: 0.0

#%%


model.setup_all()

model.solve_elastic()
model.solve_velocity()

import ufl
ν = material.ν

D = model.msh.geometry.dim
σe = es.cauchy_stress(mf.ε(model.v),ν) 
σv = 2*model.η*mf.ε(model.u) - model.p*ufl.Identity(D)
τv = 2*model.η*mf.ε(model.u)


# σv = mf.viscous_stress(mf.ε(model.u),model.p,model.η,material)


utilities.write_xdmf("outputs/unbutressed.xdmf",msh,\
                    [model.v,model.u,model.p,
                     σe,(σv),ufl.div(τv),
                     ufl.div(τv) - ufl.grad(model.p),ufl.grad(model.p)
                      ],\
                    ["v","u","p","elasticstress","viscousstress","divtauv","divu","gradp"],t=0)

#%%
tol = 0.001  # Avoid hitting the outside of the domain
ν = material.ν
y = np.linspace(tol, nondim_height - tol, 101)
x = nondim_height*np.ones_like(y)
points = np.zeros((3, 101))
points[0] = x
points[1] = y

σv = (2*model.η*mf.ε(model.u) - model.p*ufl.Identity(D))

from kraken.numerics.invariants import eigenstate2
from kraken.numerics import energy_splits as es

λ, E = eigenstate2((mf.ε(model.v)))

points_on_proc,func_vals = utilities.extract_line(points,msh,[
    es.cauchy_stress(mf.ε(model.v),ν)[0,0],
    es.cauchy_stress(mf.ε(model.v),ν)[1,1],
    σv[0,0],
    σv[1,1],
    ufl.div(σv)[0],
    ufl.div(σv)[1],
    ufl.div(es.cauchy_stress(mf.ε(model.v),ν))[0],
    ufl.div(es.cauchy_stress(mf.ε(model.v),ν))[1],
    es.cauchy_stress(mf.ε(model.v),ν)[0,1],
    σv[0,1],
    model.p,
    model.v[0],
    model.v[1]


                             ])
λ = material.λ
μ = material.μ
ρ = material.ρi
g = material.g
h = nondim_height
δ = 1-ρ/material.ρw


import matplotlib.pyplot as plt
plt.plot(func_vals[0], points_on_proc[:, 1], "r-", linewidth=1, label="σxx")
plt.plot(func_vals[1], points_on_proc[:, 1], "k-", linewidth=1, label="σzz")
plt.plot(func_vals[2], points_on_proc[:, 1], "r--", linewidth=2, label="σv_xx")
plt.plot(func_vals[3], points_on_proc[:, 1], "k--", linewidth=1.5, label="σv_zz")
# plt.plot(-func_vals[10], points_on_proc[:, 1], "r--", linewidth=1, label="p")


plt.legend()

plt.figure()
plt.plot(func_vals[4], points_on_proc[:, 1], "r-", linewidth=1, label="div(σv)_x")
plt.plot(func_vals[5], points_on_proc[:, 1], "b-", linewidth=1, label="div(σv)_y")
plt.plot(func_vals[6], points_on_proc[:, 1], "g--", linewidth=2, label="div(σe)_x")
plt.plot(func_vals[7], points_on_proc[:, 1], "k--", linewidth=2, label="div(σe)_y")
plt.legend()


plt.figure()

plt.plot(func_vals[8], points_on_proc[:, 1], "m--", linewidth=1, label="σxy")
plt.plot(func_vals[9], points_on_proc[:, 1], "c--", linewidth=1, label="σv_xy")

plt.legend()


plt.figure()

points_on_proc,funcss = utilities.extract_line(points,msh,[
    ufl.Dx(σv[0,0],0),
    ufl.Dx(σv[0,1],0),
    ufl.Dx(σe[0,0],0),
    ufl.Dx(σe[0,1],0),

                             ])


plt.plot(funcss[0], points_on_proc[:, 1], "r-", linewidth=1, label="dσv_xx/dx")
plt.plot(funcss[1], points_on_proc[:, 1], "g-", linewidth=1, label="dσv_xy/dx")
plt.plot(funcss[2], points_on_proc[:, 1], "r--", linewidth=1, label="dσe_xx/dx")
plt.plot(funcss[3], points_on_proc[:, 1], "g--", linewidth=1, label="dσe_xy/dx")

plt.plot(funcss[0]+funcss[1], points_on_proc[:, 1], "k-", linewidth=2, label="dσv_xx/dx + dσv_xy/dx")
plt.plot(funcss[2]+funcss[3], points_on_proc[:, 1], "k--", linewidth=2, label="dσe_xx/dx + dσe_xy/dx")


plt.figure()

τe = σe - ufl.tr(σe)*ufl.Identity(D)/D

points_on_proc,funcss = utilities.extract_line(points,msh,[
    τv[0,0],
    τv[0,1],
    τv[1,1],
    τe[0,0],
    τe[0,1],
    τe[1,1]

                             ])

plt.plot(funcss[0], points_on_proc[:, 1], "r-", linewidth=1, label="τv_xx")
plt.plot(funcss[1], points_on_proc[:, 1], "g-", linewidth=1, label="τv_xy")
plt.plot(funcss[2], points_on_proc[:, 1], "b-", linewidth=1, label="τv_zz")
plt.plot(funcss[3], points_on_proc[:, 1], "r--", linewidth=1, label="τe_xx")
plt.plot(funcss[4], points_on_proc[:, 1], "g--", linewidth=1, label="τe_xy")
plt.plot(funcss[5], points_on_proc[:, 1], "b--", linewidth=1, label="τe_zz")

plt.legend()


plt.figure()

plt.plot(func_vals[11], points_on_proc[:, 1], "r-", linewidth=1, label="u_x")
plt.plot(func_vals[12], points_on_proc[:, 1], "g-", linewidth=1, label="u_y")

plt.legend()