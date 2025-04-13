#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.material import Material_no_uc, Material_with_uc
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
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



# material = Material_no_uc()
material = Material_with_uc()
material.L = 1.0
material.l = 5
material.Gc = 1.0
# material.set_C1_to_one()
material.ψcrit = 1.0
# material.ν = 0.1
# material.E*= 10


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


model.setup_elastic()
model.setup_damage()

model.solve_elastic()


import ufl
pwincrack = model.pw*ufl.inner(ufl.grad(model.g), model.v)
pw2 = model.pw

utilities.write_xdmf(path + "iceberginit.xdmf",msh,\
                    [model.v,model.d,
                      mf.principal_stress(mf.ε(model.v),material.λ,material.μ),
                      mf.free_energy_plus_spectral(mf.ε(model.v),material.λ,material.μ),
                      mf.free_energy_plus_amor(mf.ε(model.v),material.λ,material.μ),
                        mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,1),
                        mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,5),
                        mf.free_energy_plus_stocek(mf.ε(model.v),material.λ,material.μ),
                      pw2,
                      mf.cauchy_stress(mf.ε(model.v),material.λ,material.μ),
                      mf.history_function(mf.ε(model.v),model.Hprev,material.λ,material.μ,material.ψcrit) \
                      ],\
                    ["v","d", "λ","spectral","amor","star1","star5","stocek","pw2","stress","history"],t=0)

#%%
tol = 0.001  # Avoid hitting the outside of the domain
y = np.linspace(-50 + tol, nondim_height - Hw - tol, 101)
x = 2e3*np.ones_like(y)
points = np.zeros((3, 101))
points[0] = x
points[1] = y

from kraken.numerics.invariants import eigenstate2

λ, E = eigenstate2((mf.ε(model.v)))

points_on_proc,func_vals = utilities.extract_line(points,msh,
    [mf.free_energy_plus_spectral(mf.ε(model.v),material.λ,material.μ),
    mf.free_energy_plus_amor(mf.ε(model.v),material.λ,material.μ),
    mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,1),
    mf.free_energy_plus_star(mf.ε(model.v),material.λ,material.μ,5),
    mf.free_energy_plus_stocek(mf.ε(model.v),material.λ,material.μ),
    mf.cauchy_stress(mf.ε(model.v),material.λ,material.μ)[0,0],
    mf.cauchy_stress(mf.ε(model.v),material.λ,material.μ)[1,1],
    model.pw,
    λ[0],
    λ[1],
    
    mf.ε(model.v)[0,0],
    mf.ε(model.v)[1,1],
    mf.ε(model.v)[0,1],

                             ])
λ = material.λ
μ = material.μ
ρ = material.ρi
g = material.g
h = nondim_height
δ = 1-ρ/material.ρw

z = points_on_proc[:, 1]
zprime = z - δ*h

σzz = ρ*g*zprime


C = g*h*ρ*(-δ*λ + 2*δ*μ - 2*μ)/(2*(λ + 2*μ))

σxx = (λ/(λ + 2 * μ))*ρ*g*z + C

pw_calc = -ρ*g*z/(1-δ)

import matplotlib.pyplot as plt
fig = plt.figure()
# plt.plot(func_vals[0], points_on_proc[:, 1], "r-", linewidth=2, label="Displacement")
plt.plot(func_vals[0], points_on_proc[:, 1], "b-", linewidth=1, label="Spectral")
# plt.plot(func_vals[1], points_on_proc[:, 1], "g-", linewidth=1, label="Amor")
# plt.plot(func_vals[2], points_on_proc[:, 1], "m-", linewidth=1, label="Star1")
# plt.plot(func_vals[3], points_on_proc[:, 1], "c-", linewidth=1, label="Star5")
# plt.plot(func_vals[4], points_on_proc[:, 1], "k-", linewidth=1, label="Stocek")
plt.figure()
plt.plot(func_vals[5], points_on_proc[:, 1], "r-", linewidth=1, label="σxx")
plt.plot(func_vals[6], points_on_proc[:, 1], "g-", linewidth=1, label="σzz")
plt.plot(func_vals[7], points_on_proc[:, 1], "b-", linewidth=1, label="pw")
plt.plot(σzz,z, "k-", linewidth=1, label="σzz calculated")
plt.plot(σxx,z, "m--", linewidth=1, label="σxx calculated")
plt.plot(pw_calc,z, "c-", linewidth=1, label="pw calculated")
plt.legend()


# plt.figure()
# plt.plot(func_vals[10], points_on_proc[:, 1], "r-", linewidth=1, label="eps00")
# plt.plot(func_vals[11], points_on_proc[:, 1], "g-", linewidth=1, label="eps11")
# plt.plot(func_vals[12], points_on_proc[:, 1], "b-", linewidth=1, label="eps01")
# plt.grid()
# plt.ylabel("y")
# plt.legend()

#%%
pw = func_vals[7][:,0]
σxx_model = func_vals[5][:,0]

F_l = h*(2*C*λ + 4*C*μ + 2*g*h*δ*λ*ρ - g*h*λ*ρ)/(2*(λ + 2*μ))

#Integrate
import numpy as np

z = points_on_proc[:, 1]
Fpw = np.trapz(pw, z)
Fσxx = np.trapz(σxx_model, z)

F_test = np.trapz(σxx, z)
print(Fpw/Fσxx)
print(Fσxx/F_test)