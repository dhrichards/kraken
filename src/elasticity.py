import numpy as np
from dolfinx import fem, default_scalar_type, la, default_real_type
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

def piola_kirchoff_stress(u, ν):
    # Spatial dimension
    d = len(u)

    λoverμ = 2 * ν / (1 - 2 * ν)

    # Identity tensor
    I = ufl.variable(ufl.Identity(d))

    # Deformation gradient
    F = ufl.variable(I + ufl.grad(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)

    # Invariants of deformation tensors
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    psi = (1.0 / 2) * (Ic - 3) - 1.0 * ufl.ln(J) + (λoverμ / 2) * (ufl.ln(J))**2
    # Stress
    # Hyper-elasticity
    P = ufl.diff(psi, F)
    return P

import copy
def deformed_normal(u,msh):
    temp_mesh = copy.copy(msh)
    temp_mesh.geometry.x[:,:msh.geometry.dim] += u.x.array.reshape((-1, msh.geometry.dim))

    return ufl.FacetNormal(temp_mesh)



def solve(msh, bc_func, material, d=None, u=None, pw=None):
    # V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    el = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type)
    
    V = fem.functionspace(msh, el)
    


    bcs = bc_func(V)

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν

    ds = ufl.Measure("ds", domain=msh)

    n = ufl.FacetNormal(msh)

    # pw = water_pressure(msh)

    if pw is None:
        pw = lambda u: water_pressure(msh,u)

    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))

    if d is None:
        d = fem.Constant(msh, default_scalar_type(0.0))
    
    g = pf.degradation(d)

    def σ(u):
        return pf.degraded_stress(u,d,ν)
        # return stress(u,ν)
        # return piola_kirchoff_stress(u, ν)
    

    # Can take u from previous timestep, or initialise to zero
    if u is None:
        u = fem.Function(V, name="elastic displacement")


    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx
    L =   C1 *( g*ufl.dot(f, v) - pw(u)*ufl.inner(ufl.grad(g), v) )* ufl.dx \
        - C1 * g * pw(u) *  ufl.dot(n, v) * ds
    

    
    F = a - L
    J = ufl.derivative(F, u, du)

    # problem = NonlinearProblem(F, u, bcs)
    
    # solver = NewtonSolver(MPI.COMM_WORLD, problem)
    # solver.convergence_criterion = "incremental"
    # solver.rtol = 1e-8
    # solver.atol = 1e-8
    # solver.max_it = 1000
    # solver.report = True

    # ksp = solver.krylov_solver
    # opts = PETSc.Options()
    # option_prefix = ksp.getOptionsPrefix()
    # opts[f"{option_prefix}ksp_type"] = "preonly"
    # # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    # opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    # ksp.setFromOptions()

    # set_log_level(LogLevel.INFO)
    # n, converged = solver.solve(u)
    # assert (converged)
    # print(f"Number of interations: {n:d}")


    # #
    F, J = fem.form(F), fem.form(J)
    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-10, max_it=10)

    # opts = PETSc.Options()


    # snes.setFromOptions()
    

    snes.getKSP().getPC().setType("lu")
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    problem = nonlinear.NonlinearPDE_SNESProblem(F, J, u, bcs)

    snes.setFunction(problem.F_mono, fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=fem.petsc.create_matrix(J), P=None)
    soln_vector = u.x.petsc_vec

    snes.solve(None, soln_vector)
    snes_converged = snes.getConvergedReason()
    ksp_converged = snes.getKSP().getConvergedReason()
    assert snes_converged > 0


    # #
    # problem = nonlinear.SNESProblem(F, u, bcs)

    # b_u = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    # J_u = fem.petsc.create_matrix(problem.a)
    # # # Create Newton solver and solve
    # solver_u_snes = PETSc.SNES().create(MPI.COMM_WORLD)
    # solver_u_snes.setType("ksponly")
    # solver_u_snes.setFunction(problem.F, b_u)
    # solver_u_snes.setJacobian(problem.J, J_u)
    # solver_u_snes.setTolerances(rtol=1.0e-9, max_it=50)
    # solver_u_snes.getKSP().setType("preonly")
    # solver_u_snes.getKSP().setTolerances(rtol=1.0e-9)
    # solver_u_snes.getKSP().getPC().setType("lu")

    # solver_u_snes.solve(None, u.x.petsc_vec)
    
    return u




def solve_no_damage(msh, bc_func, material, vh_prev, pw=None):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))




    bcs = bc_func(V)

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν

    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh)

    pw = water_pressure_static(msh)

    # if pw is None:
    #     pw = lambda u: water_pressure(msh,u)

    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))


    def σ(u):
        # return pf.degraded_stress(u,d,ν)
        return stress(u,ν)
        # return piola_kirchoff_stress(u, ν)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx
    L =   C1 *( ufl.dot(f, v) )* ufl.dx \
        - C1  * pw *  ufl.dot(n, v) * ds\
        - ufl.inner(σ(vh_prev), ε(v)) * ufl.dx
    

    
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh



