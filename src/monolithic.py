import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type, la, default_real_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.log import LogLevel, set_log_level
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
import phasefield as pf
from common import *
import basix.ufl as bufl
import nonlinear



def solve(msh, bc_func, material, Hprev=0.0, u=None, pw=None):
    # V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    vel = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type)
    sel = bufl.element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    mixed_el = bufl.mixed_element([vel, sel])
    V = fem.functionspace(msh, mixed_el)
    


    bcs = bc_func(V.sub(0))

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν
    C3 = material.C3; l = material.l

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    # pw = water_pressure(msh)

    if pw is None:
        pw = lambda u: water_pressure(msh,u)

    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))

    

    def g(d):
        return pf.degradation(d)
    
    def σ(u,d):
        # return pf.degraded_stress(u,d,ν)
        return stress(u,ν)

    def H(u):
        ψplus = pf.free_energy_plus(u,material.ν)
        # return pf.history_function(ψplus,material.ψcritstar,Hprev)
        return ψplus
    

    sol = fem.Function(V)
    u,d = ufl.split(sol)
    v,e = ufl.TestFunctions(V)


    a = ufl.inner(σ(u,d), ε(v)) * ufl.dx
    a += ((1+2*C3*l*H(u))*d*e + l**2*ufl.inner(ufl.grad(d), ufl.grad(e))) * ufl.dx
    L =   C1 *( g(d)*ufl.dot(f, v) - pw(u)*ufl.inner(ufl.grad(g(d)), v) )* ufl.dx \
        - C1 * g(d) * pw(u) *  ufl.dot(n, v) * ds
    
    L += 2*C3*l*H(u)*e * ufl.dx 
    

    
    F = a - L
    

    problem = NonlinearProblem(F, sol, bcs)
    
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.atol = 1e-6
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()

    set_log_level(LogLevel.INFO)
    n, converged = solver.solve(sol)
    assert (converged)
    print(f"Number of interations: {n:d}")


    return u,d
