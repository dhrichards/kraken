#%%

import kraken
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
from kraken.phasefield import *
from kraken.material import MaterialProperties
from dolfinx.io import XDMFFile


def left_boundary(x):
    return np.isclose(x[0], 0)

def water_pressure(msh,ρw,g):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    pw = ufl.conditional(ufl.lt(z, 0),
                         -ρw*g*z,
                         0.0)
    return pw


def move_mesh(msh,uh):
    msh.geometry.x[:,:msh.geometry.dim] += uh.x.array.reshape((-1, msh.geometry.dim))


def elasticity(msh, material):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, left_boundary)
    # boundary_dofs_x = fem.locate_dofs_topological(V, fdim, boundary_facets)
    boundary_dofs_x = fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
        

    # bc = fem.dirichletbc(np.array([0, 0], dtype=default_scalar_type), boundary_dofs_x, V)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs_x, V.sub(0))
        
    # Pull properties out
    ρi = material.ρi; ρw = material.ρw; g = material.g;
    λ = material.λ; μ = material.μ; 

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    pw = water_pressure(msh,ρw,g)

    f = fem.Constant(msh, default_scalar_type((0, -ρi*g)))

    def σ(u):
        return λ*ufl.nabla_div(u)*ufl.Identity(len(u)) + 2*μ*ε(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx - pw*ufl.dot(n, v) * ds

    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh

if __name__ == "__main__":
    L = 16e3
    H = 300

    material = MaterialProperties()
    Hw = material.ρi/material.ρw*H

    msh = mesh.create_rectangle(MPI.COMM_WORLD,
                                [np.array([-L/2, -Hw]), np.array([L/2, H-Hw])],
                                [200,50], mesh.CellType.triangle)


    uh = elasticity(msh, material)

    Q = fem.functionspace(msh, ("Lagrange", 1))
    expr = fem.Expression(water_pressure(msh,material.ρw,material.g),Q.element.interpolation_points())
    ph = fem.Function(Q)
    ph.interpolate(expr)

    with XDMFFile(MPI.COMM_WORLD, "test.xdmf", "w") as ufile_xdmf:
            ufile_xdmf.write_mesh(msh)
            ufile_xdmf.write_function(uh)
            # ufile_xdmf.write_function(ph)

    # %%
