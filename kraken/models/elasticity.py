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
from kraken.numerics import projection_tensors as pt
import basix.ufl as bufl


class ElasticitySolver:
    def __init__(self, msh, bc_func, material,dt,degradation):
        self.msh = msh
        self.material = material
        self.dt = dt
        self.g = degradation
        self.pw = lambda u: mf.water_pressure(self.msh,u,material.uc_star)


        v_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        self.V = fem.functionspace(self.msh, v_el)

        self.v_old = fem.Function(self.V, name="displacement")

        self.bcs = bc_func(self.V)

        self.ds = ufl.Measure("ds", domain=self.msh) 


        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setType("newtonls")
        opts = PETSc.Options()
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"

        self.solver.setFromOptions()

        self.solver.setTolerances(rtol=1.0e-7, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        # self.solver.getKSP().getPC().setFactorSolverType("mumps")
 



    def solve(self,v,d,u):

        ρratio = self.material.ρratio
        ν = self.material.ν
        C1 = self.material.C1


        n = ufl.FacetNormal(self.msh)

    
        pw = lambda v: self.pw(v + u*self.dt)

        f = mf.body_force(self.msh, ρratio)
        g = self.g(d)

    
        
        internal_energy = mf.degraded_free_energy(mf.ε(v), g, ν, self.material.ψcritstar) * ufl.dx

        # internal_energy = g*mf.free_energy(mf.ε(v),ν) * ufl.dx
        # internal_energy = 0.5*g*ufl.inner(mf.ε(v),mf.ε(v)) * ufl.dx

        external_energy =  self.material.C1 *( \
             g * ufl.dot(f, v) \
            + pw(v)*ufl.inner(ufl.grad(g), v)\
            # - ufl.inner(ufl.div(pw(v)*v),g)\
             )* ufl.dx \
            - self.material.C1 * g * pw(v) *  ufl.dot(n, v) * self.ds
    

        total_energy = internal_energy - external_energy
        # total_energy = self.internal_energy(v,d) - self.external_energy(v,d)

        F = ufl.derivative(total_energy,v,ufl.TestFunction(self.V))

   
        J = ufl.derivative(F,v,ufl.TrialFunction(self.V))
        # self.problem = NonlinearProblem(F, v, self.bcs)
        
        # self.solver = NewtonSolver(MPI.COMM_WORLD, self.problem)
        # self.solver.convergence_criterion = "incremental"
        # self.solver.nonlinearity_solver = "snes"
        # self.solver.rtol = 1e-7
        # self.solver.atol = 1e-7
        # self.solver.max_it = 50

        # opts = PETSc.Options()

        # opts_s_prefix = self.solver.getOptionsPrefix()
        # opts[f"{opts_s_prefix}snes_linesearch_type"] = "bt"
        # self.solver.setFromOptions()
        # # self.solver.report = True
        # # self.solver.error_on_nonconvergence = False

    
        # ksp = self.solver.krylov_solver
        
        # option_prefix = ksp.getOptionsPrefix()
        # opts[f"{option_prefix}ksp_type"] = "preonly"
        # # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
        # opts[f"{option_prefix}pc_type"] = "lu"
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
        # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
        # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
        # ksp.setFromOptions()

        # n, converged = self.solver.solve(v)

    
        # assert(converged)

        # self.problem = solvers.NonlinearPDE_SNESProblem(F, J, v, bcs=self.bcs)
        self.problem = solvers.SNESProblem(F, v, bcs=self.bcs)

       
        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)
        

        self.solver.solve(None, v.x.petsc_vec)
        self.v_old.x.array[:] = v.x.array[:]

        # return v

    def solve_linearised(self,d,u):

        v = ufl.TrialFunction(self.V)
        w = ufl.TestFunction(self.V)

        ν = self.material.ν; C1 = self.material.C1

        f = mf.body_force(self.msh, self.material.ρratio)
        g = self.g(d)
        n = ufl.FacetNormal(self.msh)
        
        pw = self.pw(self.v_old + u*self.dt)

        # a = ufl.inner(pt.degraded_stress_P(v,self.v_old,g,ν),mf.ε(w))*ufl.dx 
        a = ufl.inner(g*mf.cauchy_stress(mf.ε(v),ν),mf.ε(w))*ufl.dx
             
        L = C1*( g*ufl.dot(f,w) - pw*ufl.inner(ufl.div(w),g) )*ufl.dx
            #  + C1*g*pw(u)*ufl.inner(n,w)*ufl.ds

        problem = LinearProblem(a, L, self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        v = problem.solve()
        self.v_old.x.array[:] = v.x.array[:]
        # self.problem = NonlinearProblem(F, u, self.bcs)
        
        # self.solver = NewtonSolver(MPI.COMM_WORLD, self.problem)
        # self.solver.convergence_criterion = "incremental"
        # self.solver.rtol = 1e-7
        # self.solver.atol = 1e-7
        # self.solver.max_it = 100
        # # self.solver.report = True
        # # self.solver.error_on_nonconvergence = False

        

        # ksp = self.solver.krylov_solver
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

        # n, converged = self.solver.solve(u)

        # self.v_old.x.array[:] = u.x.array[:]

        return v







