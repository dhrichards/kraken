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





def fixed_point(msh,bcfuncs,C3,ν,l):

    


    V_u = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    V_d = fem.functionspace(msh, ("Lagrange", 1))

    bcs_u = bcfuncs[0](V_u)
    bcs_d = bcfuncs[1](V_d)

    # Define the state
    u = fem.Function(V_u, name="Displacement")
    d = fem.Function(V_d, name="Damage")



    # need upper/lower bound for the damage field
    d_lb = fem.Function(V_d, name="Lower bound")
    d_ub = fem.Function(V_d, name="Upper bound")
    d_ub.x.array[:] = 1
    d_lb.x.array[:] = 0


    total_energy = (C3*pf.degraded_free_energy(u,d,ν) + pf.γ(d,l)) * ufl.dx



    E_u = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
    E_u_u = ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
    elastic_problem = nonlinear.SNESProblem(E_u, u, bcs_u, J=E_u_u)

    b_u = la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
    J_u = fem.petsc.create_matrix(elastic_problem.a)
    # Create Newton solver and solve
    solver_u_snes = PETSc.SNES().create()
    solver_u_snes.setType("ksponly")
    solver_u_snes.setFunction(elastic_problem.F, b_u)
    solver_u_snes.setJacobian(elastic_problem.J, J_u)
    solver_u_snes.setTolerances(rtol=1.0e-9, max_it=50)
    solver_u_snes.getKSP().setType("preonly")
    solver_u_snes.getKSP().setTolerances(rtol=1.0e-9)
    solver_u_snes.getKSP().getPC().setType("lu")



    E_d = ufl.derivative(total_energy,d,ufl.TestFunction(V_d))
    E_d_d = ufl.derivative(E_d,d,ufl.TrialFunction(V_d))
    damage_problem = nonlinear.SNESProblem(E_d, d, bcs_d,J=E_d_d)


    b_d = la.create_petsc_vector(V_d.dofmap.index_map, V_d.dofmap.index_map_bs)
    J_d = fem.petsc.create_matrix(damage_problem.a)
    # Create Newton solver and solve
    solver_d_snes = PETSc.SNES().create()
    solver_d_snes.setType("vinewtonrsls")
    solver_d_snes.setFunction(damage_problem.F, b_d)
    solver_d_snes.setJacobian(damage_problem.J, J_d)
    solver_d_snes.setTolerances(rtol=1.0e-9, max_it=50)
    solver_d_snes.getKSP().setType("preonly")
    solver_d_snes.getKSP().setTolerances(rtol=1.0e-9)
    solver_d_snes.getKSP().getPC().setType("lu")
    # We set the bound (Note: they are passed as reference and not as values)
    solver_d_snes.setVariableBounds(d_lb.x.petsc_vec,d_ub.x.petsc_vec)

    for i in range(20):
        print(f"iteration {i}")
        solver_u_snes.solve(None, u.x.petsc_vec)
        solver_d_snes.solve(None, d.x.petsc_vec)

    return u,d
