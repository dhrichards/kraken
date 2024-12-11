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
from phasefield import ε
import basix.ufl as bufl
import nonlinear
import bodyforces as bf




def solve(msh, bc_func, material, d=None, u_old=None, pw=None):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    # el = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type)
    
    # V = fem.functionspace(msh, el)

    bcs = bc_func(V)

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν

    ds = ufl.Measure("ds", domain=msh)

    n = ufl.FacetNormal(msh)

    # pw = water_pressure(msh)

    if pw is None:
        pw = lambda u: bf.water_pressure(msh,u)

    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))

    if d is None:
        d = fem.Constant(msh, default_scalar_type(0.0))
    
    g = pf.degradation(d)

    # Can take u from previous timestep, or initialise to zero
    
    u = fem.Function(V, name="elastic displacement")
    if u_old is not None:
        u.x.array[:] = u_old.x.array[:]


    internal_energy = pf.degraded_free_energy(pf.ε(u),d,ν,material.ψcritstar) * ufl.dx
    # internal_energy = (pf.degradation(d)*free_energy(u,ν) + (1/C3)*pf.γ(d,l)) * ufl.dx

    external_energy =  C1 *( g*ufl.dot(f, u) - pw(u)*ufl.inner(ufl.grad(g), u) )* ufl.dx \
        - C1 * g * pw(u) *  ufl.dot(n, u) * ds
    

    total_energy = internal_energy - external_energy

    F = ufl.derivative(total_energy,u,ufl.TestFunction(V))

    # def σ(u):
    #     return pf.degraded_stress(u,d,ν)

    # v = ufl.TestFunction(V)
    # du = ufl.TrialFunction(V)

    # a = ufl.inner(σ(u), ε(v)) * ufl.dx
    # L =   C1 *( g*ufl.dot(f, v) - pw(u)*ufl.inner(ufl.grad(g), v) )* ufl.dx \
    #     - C1 * g * pw(u) *  ufl.dot(n, v) * ds
    
    # F = a - L
    # J = ufl.derivative(F, u, du)


    problem = NonlinearProblem(F, u, bcs)
    
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-8
    solver.max_it = 100
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

    n, converged = solver.solve(u)
    assert (converged)
    


    
    return u




def solve_no_damage(msh, bc_func, material, pw=None):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    


    bcs = bc_func(V)

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν

    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh)

    pw = bf.water_pressure_static(msh)

    # if pw is None:
    #     pw = lambda u: bf.water_pressure(msh,u)


    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))


    def σ(u):
        # return pf.degraded_stress(u,d,ν)
        return pf.stress(u,ν)
        # return piola_kirchoff_stress(u, ν)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(σ(u), ε(v)) * ufl.dx
    L =   C1 *( ufl.dot(f, v) )* ufl.dx \
        - C1  * pw *  ufl.dot(n, v) * ds\
    

    
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh
    # F = a - L
    # problem = NonlinearProblem(F, u, bcs)

    # solver = NewtonSolver(MPI.COMM_WORLD, problem)
    # solver.convergence_criterion = "incremental"
    # solver.rtol = 1e-8
    # solver.atol = 1e-8
    # solver.max_it = 100
    # solver.report = True

    

    # ksp = solver.krylov_solver
    # opts = PETSc.Options()
    # option_prefix = ksp.getOptionsPrefix()
    # opts[f"{option_prefix}ksp_type"] = "preonly"
    # # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    # opts[f"{option_prefix}pc_type"] = "lu"
    # # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    # ksp.setFromOptions()

    # n, converged = solver.solve(u)
    # assert (converged)
    


    
    # return u



