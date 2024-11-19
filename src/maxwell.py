#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from elasticity import elasticity, water_pressure, elasticity_no_damage, stress_nondim
from material import MaterialProperties
import invariants

def left_boundary(x):
    return np.isclose(x[0], 0)


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
                            [200,50], mesh.CellType.triangle)


uh = elasticity(msh,material,0.0)



σ = stress_nondim(uh,material.ν)
λ,E = invariants.eigenstate(σ)



Q = fem.functionspace(msh, ("Lagrange", 1))
expr = fem.Expression(water_pressure(msh,material.ρw,material.g),Q.element.interpolation_points())
ph = fem.Function(Q)
ph.interpolate(expr)

λ1 = fem.Function(Q)
λ1.interpolate(fem.Expression(λ[0],Q.element.interpolation_points()))

λ2 = fem.Function(Q)
λ2.interpolate(fem.Expression(λ[1],Q.element.interpolation_points()))



from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "testnondim.xdmf", "w") as ufile_xdmf:
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(uh)
        # ufile_xdmf.write_function(ph)
        # ufile_xdmf.write_function(λ1)

# %%
