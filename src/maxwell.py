#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
import elasticity
from material import MaterialProperties
import invariants
from boundaryconditions import get_zero_bc
import stokes
import poisson
import phasefield

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


def move_mesh(msh,uh):
    msh.geometry.x[:,:msh.geometry.dim] += uh.x.array.reshape((-1, msh.geometry.dim))


true_length = 16e3
true_height = 300

material = MaterialProperties()

nondim_length = true_length/material.L
nondim_height = true_height/material.L


Hw = material.ρi/material.ρw*nondim_height

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
                            [50,5], mesh.CellType.triangle)


# 
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []

bc = clamped_bc
# vh = elasticity.solve(msh,material,symm_bc,0.0)

vh = elasticity.solve(msh,material,bc)


σ = elasticity.stress(vh,material.ν)

uhp = poisson.velocity(msh,vh,bc,material)
uh, ph = stokes.solve_no_damage(msh, vh, bc, material)

dh,H = phasefield.solve(msh,vh,material,0.0)

λ,E = invariants.eigenstate(σ)



Q = fem.functionspace(msh, ("Lagrange", 1))
# expr = fem.Expression(elasticity.water_pressure(msh,material.ρw,material.g),Q.element.interpolation_points())
# ph = fem.Function(Q)
# ph.interpolate(expr)

λ1 = fem.Function(Q)
λ1.interpolate(fem.Expression(λ[0],Q.element.interpolation_points()))

λ2 = fem.Function(Q)
λ2.interpolate(fem.Expression(λ[1],Q.element.interpolation_points()))

V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
uhh = fem.Function(V)
uhh.interpolate(uh)

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "displacement.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(vh)
with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(uhh)

with XDMFFile(MPI.COMM_WORLD, "velocitypoisson.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(uhp)
