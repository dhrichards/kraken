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


true_length = 16e3
true_height = 300

material = MaterialProperties()

nondim_length = true_length/material.L
nondim_height = true_height/material.L


Hw = material.ρi/material.ρw*nondim_height

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
                            [200,50], mesh.CellType.triangle)


material.set_l_from_mesh(msh)
# 
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []

bc = symm_bc
# vh = elasticity.solve(msh,material,symm_bc,0.0)

vh = elasticity.solve(msh,material,bc)


σ = elasticity.stress(vh,material.ν)

# uhp = poisson.velocity(msh,vh,bc,material)
# uh, ph = stokes.solve_no_damage(msh, vh, bc, material)

ψ = phasefield.free_energy_plus(vh,material.ν)

dh,H = phasefield.solve(msh,vh,material,0.0)

from invariants import matrix_function
from common import *
λ,E = invariants.eigenstate(matrix_function((ε(vh)),phasefield.positive_part))

test = phasefield.positive_part(ufl.tr(ε(vh)))


Q = fem.functionspace(msh, ("Lagrange", 1))
# expr = fem.Expression(elasticity.water_pressure(msh,material.ρw,material.g),Q.element.interpolation_points())
# ph = fem.Function(Q)
# ph.interpolate(expr)

λ1 = fem.Function(Q)
λ1.interpolate(fem.Expression(λ[0],Q.element.interpolation_points()))

λ2 = fem.Function(Q)
λ2.interpolate(fem.Expression(λ[1],Q.element.interpolation_points()))

ttest = fem.Function(Q)
ttest.interpolate(fem.Expression(test,Q.element.interpolation_points()))

V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
uhh = fem.Function(V)

ψh = fem.Function(Q)
expr = fem.Expression(ψ,Q.element.interpolation_points())
ψh.interpolate(expr)

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "displacement.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(vh)

# uhh.interpolate(uh)
# with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as ufile_xdmf:
#         ufile_xdmf.write_mesh(msh)
#         ufile_xdmf.write_function(uhh)

# with XDMFFile(MPI.COMM_WORLD, "velocitypoisson.xdmf", "w") as ufile_xdmf:
#         ufile_xdmf.write_mesh(msh)
#         ufile_xdmf.write_function(uhp)

with XDMFFile(MPI.COMM_WORLD, "freeenergy.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(ψh)

with XDMFFile(MPI.COMM_WORLD, "phasefield.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(dh)

with XDMFFile(MPI.COMM_WORLD, "eigenstate1.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(λ1)

with XDMFFile(MPI.COMM_WORLD, "eigenstate2.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(λ2)

with XDMFFile(MPI.COMM_WORLD, "test.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(ttest)
