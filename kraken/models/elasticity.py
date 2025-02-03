from dolfinx import fem, default_scalar_type, la, default_real_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
from kraken.numerics import maths_functions as mf
from kraken.numerics.maths_functions import ε
from kraken.numerics import solvers
import basix.ufl as bufl


class ElasticitySolver:
    def __init__(self, msh, bc_func, material,dt):
        self.msh = msh
        self.material = material
        self.dt = dt

        v_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        self.V = fem.functionspace(self.msh, v_el)

        self.v_old = fem.Function(self.V, name="displacement")

        self.bcs = bc_func(self.V)

    def solve(self,v,d,u):

        ρratio = self.material.ρratio
        ν = self.material.ν

        ds = ufl.Measure("ds", domain=self.msh)

        n = ufl.FacetNormal(self.msh)

    
        pw = lambda v: mf.water_pressure(self.msh,v+u*self.dt)

        f = mf.body_force(self.msh, ρratio)
        g = mf.degradation(d)

    

        internal_energy = mf.degraded_free_energy(mf.ε(v),g,ν,self.material.ψcritstar) * ufl.dx
        # internal_energy = (pf.degradation(d)*free_energy(u,ν) + (1/C3)*pf.γ(d,l)) * ufl.dx

        external_energy =  self.material.C1 *( g*ufl.dot(f, v) - pw(v)*ufl.inner(ufl.grad(g), v) )* ufl.dx \
            - self.material.C1 * g * pw(v) *  ufl.dot(n, v) * ds
        

        total_energy = internal_energy - external_energy
        # total_energy = self.internal_energy(v,d) - self.external_energy(v,d)

        F = ufl.derivative(total_energy,v,ufl.TestFunction(self.V))
        # J = ufl.derivative(F,v,ufl.TrialFunction(self.V))



        self.problem = NonlinearProblem(F, v, self.bcs)
        
        self.solver = NewtonSolver(MPI.COMM_WORLD, self.problem)
        self.solver.convergence_criterion = "incremental"
        # self.solver.nonlinearity_solver = "snes"
        self.solver.rtol = 1e-7
        self.solver.atol = 1e-7
        self.solver.max_it = 50
        # self.solver.report = True
        # self.solver.error_on_nonconvergence = False

        

        ksp = self.solver.krylov_solver
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

        n, converged = self.solver.solve(v)

    
        # assert(converged)

        # self.problem = solvers.NonlinearPDE_SNESProblem(F, J, v, bcs=self.bcs)

        # self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setType("ksponly")
        # self.solver.setFunction(self.problem.F_mono, fem.petsc.create_vector(fem.form(F)))
        # self.solver.setJacobian(self.problem.J_mono, fem.petsc.create_matrix(fem.form(J)),P=None)
        # self.solver.setTolerances(rtol=1.0e-7, max_it=50)
        # self.solver.getKSP().setType("preonly")
        # self.solver.getKSP().setTolerances(rtol=1.0e-7)
        # self.solver.getKSP().getPC().setType("lu")

        # self.solver.solve(None, v.x.petsc_vec)

        # return v

    def solve_linearised(self,u,d):

        v = ufl.TestFunction(self.V)

        ρratio = self.material.ρratio; ν = self.material.ν
        C1 = self.material.C1

        f = mf.body_force(self.msh, self.material.ρratio)
        g = mf.degradation(d)
        n = ufl.FacetNormal(self.msh)
        
        pw = lambda u: mf.water_pressure(self.msh,u)# -u[self.msh.geometry.dim-1]


        F = ( ufl.inner(mf.degraded_stress_P(u,self.v_old,d,ν),mf.ε(v)) \
             - C1*g*ufl.inner(f,v) \
             + C1*pw(u)*ufl.inner(ufl.grad(g),v) ) * ufl.dx \
             + C1*g*pw(u)*ufl.inner(n,v)*ufl.ds


        
        # a, L = ufl.lhs(F), ufl.rhs(F)

        # problem = LinearProblem(a, L, self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        # u = problem.solve()
        self.problem = NonlinearProblem(F, u, self.bcs)
        
        self.solver = NewtonSolver(MPI.COMM_WORLD, self.problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-7
        self.solver.atol = 1e-7
        self.solver.max_it = 100
        # self.solver.report = True
        # self.solver.error_on_nonconvergence = False

        

        ksp = self.solver.krylov_solver
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

        n, converged = self.solver.solve(u)

        self.v_old.x.array[:] = u.x.array[:]

        # return u







