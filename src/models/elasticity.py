from dolfinx import fem, default_scalar_type, la, default_real_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
from numerics import maths_functions as mf
from numerics.maths_functions import ε
import basix.ufl as bufl


class ElasticitySolver:
    def __init__(self, msh, bc_func, material):
        self.msh = msh
        self.material = material

        v_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        self.V = fem.functionspace(self.msh, v_el)

        self.bcs = bc_func(self.V)

    def solve(self,v,d):

        ρratio = self.material.ρratio
        ν = self.material.ν

        ds = ufl.Measure("ds", domain=self.msh)

        n = ufl.FacetNormal(self.msh)

    
        pw = lambda u: mf.water_pressure(self.msh,u)

        f = mf.body_force(self.msh, ρratio)
        g = mf.degradation(d)

        
    


        internal_energy = mf.degraded_free_energy(mf.ε(v),d,ν,self.material.ψcritstar) * ufl.dx
        # internal_energy = (pf.degradation(d)*free_energy(u,ν) + (1/C3)*pf.γ(d,l)) * ufl.dx

        external_energy =  self.material.C1 *( g*ufl.dot(f, v) - pw(v)*ufl.inner(ufl.grad(g), v) )* ufl.dx \
            - self.material.C1 * g * pw(v) *  ufl.dot(n, v) * ds
        

        total_energy = internal_energy - external_energy

        F = ufl.derivative(total_energy,v,ufl.TestFunction(self.V))



        self.problem = NonlinearProblem(F, v, self.bcs)
        
        self.solver = NewtonSolver(MPI.COMM_WORLD, self.problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-8
        self.solver.atol = 1e-8
        self.solver.max_it = 100
        self.solver.report = True

        

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

        n, converged = self.solver.solve(v)
        assert(converged)

        # return v




