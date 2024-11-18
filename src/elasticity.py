import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from phasefield import *

def left_boundary(x):
    return np.isclose(x[0], 0)

def water_pressure(msh,ρw=1.0,g=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    pw = ufl.conditional(ufl.lt(z, 0),
                         -ρw*g*z,
                         0.0)
    return pw



def elasticity(msh, dh, material):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, left_boundary)
    # boundary_dofs_x = fem.locate_dofs_topological(V, fdim, boundary_facets)
    boundary_dofs_x = fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
        

    # bc = fem.dirichletbc(np.array([0, 0], dtype=default_scalar_type), boundary_dofs_x, V)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs_x, V.sub(0))
        
    # Pull properties out
    ρi = material.ρi; ρw = material.ρw; C1 = material.C1
    ν = material.ν

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    pw = water_pressure(msh)

    f = fem.Constant(msh, default_scalar_type((0, -ρi/ρw)))

    g = degradation(dh)

    def σ(u):
        return degraded_stress(u,dh,ν)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx
    L =   C1 *( g*ufl.dot(f, v) + pw*g*ufl.div(v) )* ufl.dx \
        - C1 * g * pw *  ufl.dot(n, v) * ds

    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh