from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
import ufl



def transport(bcfunc, uh, fh, dt, k = 1e-4):
    # Solve transport equation with some artificial diffusion
    F = f.function_space

    bc = bcfunc(F)

    f = ufl.TrialFunction(F)
    g = ufl.TestFunction(F)

    a = ( ufl.inner(f,g) \
        + dt*ufl.inner(ufl.inner(uh, ufl.grad(f)), g) \
        + dt*k*ufl.inner(ufl.grad(f), ufl.grad(g)) )*ufl.dx
    
    L = ufl.inner(fh,g)*ufl.dx

    problem = fem.LinearProblem(a, L, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    fh = problem.solve()

    return fh



def move_z(msh, uh, dt, acc=0.0, k = 1e-4):
    # Move mesh in z direction according to velocity field
    # This is a simple first order Euler scheme

    # Get mesh geometry
    x = ufl.SpatialCoordinate(msh)
    z_old = x[-1]
    D = msh.geometry.dim

    Z = fem.FunctionSpace(msh, ("Lagrange", 1))

    z = ufl.TrialFunction(Z)
    v = ufl.TestFunction(Z)

    us = uh[:D-1]; w = uh[-1]

    grad_s = lambda u: ufl.as_vector([ufl.grad(u)[i] for i in range(D-1)])

    ds = ufl.Measure("ds", domain=msh)




    a = (ufl.inner(z, v) + dt*ufl.inner(us, grad_s(z)))*ds \
        + k*ufl.inner(ufl.grad(z), ufl.grad(v))*ufl.dx
    
    L = ufl.inner(dt*(w+acc), v)*ds

    problem = fem.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dz = problem.solve()

    return dz

