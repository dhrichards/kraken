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


def solve(msh, material, bc_func, d=0.0, u=None):
    # V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    el = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type)
    
    V = fem.functionspace(msh, el)
    


    bcs = bc_func(V)

    # Pull properties out
    ρi = material.ρi; ρw = material.ρw; C1 = material.C1
    ν = material.ν

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    # pw = water_pressure(msh)

    def pw(u):
        return water_pressure(msh,u,material)

    f = fem.Constant(msh, default_scalar_type((0, -ρi/ρw)))

    g = pf.degradation(d)

    def σ(u):
        return pf.degraded_stress(u,d,ν)
        # return stress_nondim(u,ν)
    

    # Because of the degraded stress the problem is non linear
    if u is None: # Get initial guess from linear solve with no damage
        # u = elasticity_no_damage(msh, material, bc_func)
        u = fem.Function(V)


    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx
    L =   C1 *( g*ufl.dot(f, v) + g*pw(u)*ufl.div(v) )* ufl.dx \
        - C1 * g * pw(u) *  ufl.dot(n, v) * ds
    

    
    F = a - L
    J = ufl.derivative(F, u, du)

    problem = NonlinearProblem(F, u, bcs)
    
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    # solver.rtol = 1e-6
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    # ksp.setFromOptions()

    set_log_level(LogLevel.INFO)
    n, converged = solver.solve(u)
    assert (converged)
    print(f"Number of interations: {n:d}")


    # #
    # F, J = fem.form(F), fem.form(J)
    # snes = PETSc.SNES().create(MPI.COMM_WORLD)
    # snes.setTolerances(rtol=1.0e-10, max_it=10)

    # # opts = PETSc.Options()


    # # snes.setFromOptions()
    

    # snes.getKSP().getPC().setType("none")
    # snes.getKSP().setType("gmres")
    # # snes.getKSP().getPC().setFactorSolverType("mumps")

    # problem = nonlinear.NonlinearPDE_SNESProblem(F, J, u, bcs)

    # snes.setFunction(problem.F_mono, fem.petsc.create_vector(F))
    # snes.setJacobian(problem.J_mono, J=fem.petsc.create_matrix(J), P=None)
    # soln_vector = u.x.petsc_vec

    # snes.solve(None, soln_vector)
    # snes_converged = snes.getConvergedReason()
    # ksp_converged = snes.getKSP().getConvergedReason()
    # if snes_converged < 1 or ksp_converged < 1:
    #     info(f"SNES converged reason: {snes_converged}")
    #     info(f"KSP converged reason: {ksp_converged}")


    #
    # problem = nonlinear.SNESProblem(F, u, bc)

    # b_u = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    # J_u = fem.petsc.create_matrix(problem.a)
    # # Create Newton solver and solve
    # solver_u_snes = PETSc.SNES().create()
    # solver_u_snes.setType("ksponly")
    # solver_u_snes.setFunction(problem.F, b_u)
    # solver_u_snes.setJacobian(problem.J, J_u)
    # solver_u_snes.setTolerances(rtol=1.0e-9, max_it=50)
    # solver_u_snes.getKSP().setType("gmres")
    # solver_u_snes.getKSP().setTolerances(rtol=1.0e-9)
    # solver_u_snes.getKSP().getPC().setType("none")
    return u


def solve_no_damage(msh, material, bc_func):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    bcs = bc_func(V)

    # Pull properties out
    ρi = material.ρi; ρw = material.ρw; C1 = material.C1
    ν = material.ν

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    # This doesn't actually work - need displacement as part of water pressure
    # to give correct solution for ice berg
    pw = water_pressure_static(msh)


    f = fem.Constant(msh, default_scalar_type((0, -ρi/ρw)))

    def σ(u):
        return stress(u,ν)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx

    L =   C1 * ufl.inner(f, v) * ufl.dx \
        - C1 * pw *  ufl.inner(n, v) * ds

    problem = LinearProblem(a, L, bcs=bcs, 
                            # petsc_options={"ksp_type": "fgmres", "pc_type": "none"})
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh
