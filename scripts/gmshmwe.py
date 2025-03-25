#%%
from dolfinx import fem, io, default_scalar_type, mesh, default_real_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import icebergmesh
from material import Material_no_uc
from mpi4py import MPI
import numpy as np
import boundaryconditions as bc


true_length = 4e3
true_height = 300

material = Material_no_uc()
material.τ = 3600*24
# material.L = true_height    
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.l = 10.0/material.L

cell_size = material.l/2.1
nz = int(nondim_height/cell_size)
nx = int(nondim_length/cell_size/2)


Hw = material.ρi/material.ρw*nondim_height
model = icebergmesh.create_iceberg_mesh(true_length, true_height, material)
msh,ct,ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)


V = fem.functionspace(msh, ("Lagrange", 1))

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


# facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, left_boundary)
# dofs = fem.locate_dofs_topological(V, entity_dim=1, entities=facets)
# bc = fem.dirichletbc(value=default_scalar_type(1.0), dofs=dofs, V=V)
# left_bc = bc.get_bc(V, left_boundary, default_scalar_type(1.0))
# right_bc = bc.get_bc(V, right_boundary, default_scalar_type(0.0))

# n = ufl.FacetNormal(msh)
# ds = ufl.Measure("ds", domain=msh)

# u = ufl.TrialFunction(V)
# v = ufl.TestFunction(V)

# f = fem.Constant(msh, 0.0)
# a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
# L = ufl.inner(f, v) * ufl.dx + ufl.inner(f, v) * ds

# problem = LinearProblem(a, L, bcs=[left_bc,right_bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh = problem.solve()

import elasticity

bcs = lambda V: [bc.get_zero_bc(V, left_boundary, default_scalar_type)]


# uh = elasticity.solve_no_damage(msh, bcs, material)
uh = elasticity.solve(msh, bcs, material)

with io.XDMFFile(MPI.COMM_WORLD, "output/gmshmwe.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh)
