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
    return np.isclose(x[1], -Hw)

def crack(x):
    x_c = nondim_length/2 - nondim_height
    l = material.lstar
    return (x[0]<0.75*l)*(x[1]>0.5*(nondim_height-Hw))


def fixed(x):
    return x[0]>0.4*nondim_height


## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 16e3
true_height = 300

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



material = Params_no_uc()
# material = Material_with_uc()
material.L = 1.0
material.l = 10
material.Gc = 1.0
# material.set_C1_to_one()
material.ψcrit = 1.0
# material.ν = 0.1
# material.E*= 10
material.patm=0.0


# material.L = true_height
# material.τ = 3600*24
nondim_length = true_length/material.L
nondim_height = true_height/material.L

Hw = material.ρi/material.ρw*nondim_height


# filename = "icebergL" + str(int(true_length/1e3)) + "l" + str(int(material.l*material.L)) + ".xdmf"

# # msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
# with io.XDMFFile(MPI.COMM_WORLD,"../meshes/" + filename,"r") as infile:
#     msh = infile.read_mesh()
1
# msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
cell_size = material.lstar/2.1


# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
#                             [20,100], mesh.CellType.quadrilateral)

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([-nondim_length/2, 0]), np.array([nondim_length/2, 1])],
                            [20,100], mesh.CellType.quadrilateral)

msh.geometry.x[:,1] = msh.geometry.x[:,1]**1.0
msh.geometry.x[:,1] *= nondim_height
msh.geometry.x[:,1] -= Hw



clamped_both = lambda V: [bc.get_zero_bc(V, left_boundary),
                            bc.get_zero_bc(V, right_boundary)]

clamped_bc = lambda V: [bc.get_zero_bc(V, left_boundary)]
symm_bc = lambda V: [
    # bc.get_zero_bc(V.sub(0), left_boundary),
                    #  bc.get_zero_bc(V.sub(1), bottom_boundary)
                     ]
no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0),
                  bc.internal_bc(V, crack, 1.0)]
# bc_d = lambda V: [bc.internal_bc(V, lambda x: x<(x_change+0.1), 0.0)]

cliff_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1), bottom_boundary)]

model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], material, 
                               dt = 1)#g = lambda d: mf.degradation_Lo2023(d,0.05))

model.bounded = False
# model.setup_elastic()
# model.setup_damage()
# model.pw = 0.0


#%%


model.setup_all()

model.solve_elastic()
model.solve_velocity()

import ufl
pwincrack = model.p*ufl.inner(ufl.grad(model.g), model.v)
# pw2 = model.p

# utilities.write_xdmf(path + "iceberginit.xdmf",msh,\
#                     [model.v,model.d,
#                       mf.principal_stress(mf.ε(model.v),ν),
#                       mf.free_energy_plus_spectral(mf.ε(model.v),ν),
#                       mf.free_energy_plus_amor(mf.ε(model.v),ν),
#                         mf.free_energy_plus_star(mf.ε(model.v),ν,1),
#                         mf.free_energy_plus_star(mf.ε(model.v),ν,5),
#                         mf.free_energy_plus_stocek(mf.ε(model.v),ν),
#                       pw2,
#                       mf.cauchy_stress(mf.ε(model.v),ν),
#                       mf.history_function(mf.ε(model.v),model.Hprev,ν,material.ψcrit) \
#                       ],\
#                     ["v","d", "λ","spectral","amor","star1","star5","stocek","pw2","stress","history"],t=0)

#%%
tol = 0.001  # Avoid hitting the outside of the domain
ν = material.ν
y = np.linspace(-50 + tol, nondim_height - Hw - tol, 101)
x = 2e3*np.ones_like(y)
points = np.zeros((3, 101))
points[0] = x
points[1] = y

from kraken.numerics.invariants import eigenstate2
from kraken.numerics import energy_splits as es

λ, E = eigenstate2((mf.ε(model.v)))

points_on_proc,func_vals = utilities.extract_line(points,msh,[
    es.cauchy_stress(mf.ε(model.v),ν)[0,0],
    es.cauchy_stress(mf.ε(model.v),ν)[1,1],
    mf.ε(model.u)[0,0],
    mf.ε(model.u)[1,1],
    mf.ε(model.u)[0,1],
    model.p,

                             ])
λ = material.λ
μ = material.μ
ρ = material.ρi
g = material.g
h = nondim_height
δ = 1-ρ/material.ρw

z = points_on_proc[:, 1]
zprime = z - δ*h

σzz = ρ*g*zprime/μ 


C = g*h*ρ*(-δ*λ + 2*δ*μ - 2*μ)/(2*(λ + 2*μ))

σxx = (λ/(λ + 2 * μ))*ρ*g*z + C
σxx /= μ

pw_calc = -ρ*g*z/(1-δ)

import matplotlib.pyplot as plt
plt.plot(func_vals[0], points_on_proc[:, 1], "r-", linewidth=1, label="σxx")
plt.plot(func_vals[1], points_on_proc[:, 1], "g-", linewidth=1, label="σzz")
plt.plot(σzz,z, "k-", linewidth=1, label="σzz calculated")
plt.plot(σxx,z, "m--", linewidth=1, label="σxx calculated")
plt.plot(-func_vals[5], z, "b-", linewidth=1, label="p")
plt.legend()


plt.figure()
plt.plot(func_vals[2], points_on_proc[:, 1], "r-", linewidth=1, label="εxx")
plt.plot(func_vals[3], points_on_proc[:, 1], "g-", linewidth=1, label="εzz")
plt.plot(func_vals[4], points_on_proc[:, 1], "b-", linewidth=1, label="εxz")
plt.plot(func_vals[5],z, "k-", linewidth=1, label="p")


