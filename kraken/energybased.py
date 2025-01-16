from dolfinx import fem, la, default_real_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
import phasefield as pf
from kraken import bodyforces as bf
import basix.ufl as bufl
import nonlinear






def fixed_point(msh,bcfuncs,material, d_lb=None, u_old=None, pw=None, max_its = 100, tol=1e-4):

    

    vel = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type)
    
    V_u = fem.functionspace(msh, vel)
    
    # V_u = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    d_el = bufl.element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    V_d = fem.functionspace(msh, d_el)

    bcs_u = bcfuncs[0](V_u)
    bcs_d = bcfuncs[1](V_d)

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν
    C3 = material.C3; l = material.l; ψcrit = material.ψcritstar



    # Define the state
   
    u = fem.Function(V_u, name="Displacement")
    
    d = fem.Function(V_d, name="Damage")
    
    if d_lb is not None:
        d.x.array[:] = d_lb.x.array[:]

    if u_old is not None:
        u.x.array[:] = u_old.x.array[:]



    # need upper/lower bound for the damage field
    if d_lb is None:
        d_lb = fem.Function(V_d, name="Lower bound")
        d_lb.x.array[:] = 0
    d_ub = fem.Function(V_d, name="Upper bound")
    d_ub.x.array[:] = 1
    

    g = lambda d: pf.degradation(d)
    f = bf.body_force(msh, ρratio)
    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh)
    pw = lambda u: bf.water_pressure(msh, u)
    



    internal_energy = (pf.degraded_free_energy(pf.ε(u),d,ν,ψcrit) + (1/C3)*pf.γ(d,l)) * ufl.dx
    # internal_energy = (pf.degradation(d)*free_energy(u,ν) + (1/C3)*pf.γ(d,l)) * ufl.dx

    external_energy =  C1 *( ufl.dot(f, u) - pw(u)*ufl.inner(ufl.grad(g(d)), u) )* ufl.dx \
        - C1 * g(d) * pw(u) *  ufl.dot(n, u) * ds
    

    total_energy = internal_energy - external_energy

    E_u = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
    E_u_u = ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
    # elastic_problem = nonlinear.SNESProblem(E_u, u, bcs_u, J=E_u_u)

    # b_u = la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
    # J_u = fem.petsc.create_matrix(elastic_problem.a)
    # # Create Newton solver and solve
    # solver_u_snes = PETSc.SNES().create(MPI.COMM_WORLD)
    # solver_u_snes.setType("ksponly")
    # solver_u_snes.setTolerances(rtol=1.0e-9, max_it=50)
    # solver_u_snes.getKSP().setType("cg")
    # solver_u_snes.getKSP().setTolerances(rtol=1.0e-9)
    # solver_u_snes.getKSP().getPC().setType("gamg")


    # elastic_problem = nonlinear.NonlinearPDE_SNESProblem(
    #     fem.form(E_u), fem.form(E_u_u), u, bcs=bcs_u)
    # solver_u_snes.setFunction(elastic_problem.F_mono, fem.petsc.create_vector(fem.form(E_u)))
    # solver_u_snes.setJacobian(elastic_problem.J_mono, fem.petsc.create_matrix(fem.form(E_u_u)),P=None)
    
    elastic_problem = NonlinearProblem(E_u, u, bcs_u)
    
    solver_u = NewtonSolver(MPI.COMM_WORLD, elastic_problem)
    solver_u.convergence_criterion = "incremental"
    solver_u.rtol = 1e-8
    solver_u.atol = 1e-8
    solver_u.max_it = 100
    solver_u.report = True

    ksp = solver_u.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()

    # set_log_level(LogLevel.INFO)
    




    E_d = ufl.derivative(internal_energy,d,ufl.TestFunction(V_d))
    E_d_d = ufl.derivative(E_d,d,ufl.TrialFunction(V_d))
    damage_problem = nonlinear.NonlinearPDE_SNESProblem(
        fem.form(E_d), fem.form(E_d_d), d, bcs=bcs_d)
    


    # Create Newton solver and solve
    solver_d_snes = PETSc.SNES().create(MPI.COMM_WORLD)
    solver_d_snes.setType("vinewtonrsls")
    solver_d_snes.setFunction(damage_problem.F_mono, fem.petsc.create_vector(fem.form(E_d)))
    solver_d_snes.setJacobian(damage_problem.J_mono, fem.petsc.create_matrix(fem.form(E_d_d)),P=None)
    solver_d_snes.setTolerances(rtol=1.0e-9, max_it=50)
    solver_d_snes.getKSP().setType("preonly")
    solver_d_snes.getKSP().setTolerances(rtol=1.0e-9)
    solver_d_snes.getKSP().getPC().setType("lu")
    solver_d_snes.getKSP().getPC().setFactorSolverType("mumps")
    # We set the bound (Note: they are passed as reference and not as values)
    solver_d_snes.setVariableBounds(d_lb.x.petsc_vec,d_ub.x.petsc_vec)


    u,d = minimisation(solver_u,solver_d_snes,u,d,max_its,tol)
    return u,d



def minimisation(solver_u,solver_d, u, d, max_its=100, tol=1e-4):

    L2_old = 0.0
    for i in range(max_its):

        # solver_u.solve(None,u.x.petsc_vec)
        n, converged = solver_u.solve(u)
        solver_d.solve(None,d.x.petsc_vec)
        # solver_d.solve(d)

        # print(solver_u.getConvergedReason())
        # assert solver_u.getConvergedReason() > 0
        assert (converged)

        L2_ = ufl.inner(d,d)*ufl.dx
        L2_rank = fem.assemble_scalar(fem.form(L2_))
        L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

        error_L2 = np.abs(L2 - L2_old)
        
        
        print(f"iteration {i}, error {error_L2}")

        if i>0 and error_L2 < tol:
        # if i > 35:
            break

        L2_old = L2

    return u,d

        