from dolfinx import fem, nls
import ufl
import phasefield as pf
from common import *
from mpi4py import MPI

def velocity_linear(msh,σ,bc_func):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    bcs = bc_func(V)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ε(u), ε(v)) * ufl.dx
    L = -ufl.inner(ufl.dev(σ), ε(v)) * ufl.dx


    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()

    return uh


def velocity(msh,vh,bc_func,material,d=0.0,u=None):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    bcs = bc_func(V)

    if u is None:
        u = fem.Function(V)

    v = ufl.TestFunction(V)

    C2 = material.C2

    def η(u):
        return viscosity(u, material.n, 1.e-8)
    
    σ = stress(vh,material.ν)
    
    g = pf.degradation(d)

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    pw = water_pressure(msh,vh,material)

    a = g * η(u) * ufl.inner(ε(u), ε(v)) * ufl.dx
    L = ufl.inner(ufl.dev(σ), ε(v)) * ufl.dx\
        - (-pw - ufl.tr(σ)/3)*ufl.inner(n, v)*ds

    F = a - L

    problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs)

    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    n, converged = solver.solve(u)
    assert (converged)


    return u
