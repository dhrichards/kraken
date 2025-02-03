import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type, nls
from kraken.numerics import maths_functions as mf
from kraken.numerics.maths_functions import ε
from kraken.numerics import advection_numerics, solvers
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np


class DamageSolver:
    def __init__(self, msh, bc_func, material, w=lambda d: d**2):
        self.msh = msh
        self.material = material
        self.w = w
        #c0 = 4*int_0^1 sqrt(w(s))ds
        self.c0 = np.trapz(np.sqrt(self.w(np.linspace(0,1,100))),np.linspace(0,1,100))

        d_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        self.V = fem.functionspace(self.msh, d_el)
        


        self.bcs = bc_func(self.V)

    def solve(self,v,Hprev,d_old):

        H = mf.history_function(ε(v),Hprev,self.material.ν,self.material.ψcritstar)

        C3 = self.material.C3; l = self.material.l

        d, g = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        

        # only for w=d**2
        F = (ufl.inner(d,g) + l**2*ufl.inner(ufl.grad(d), ufl.grad(g)) \
                + C3*l*mf.deriv_deg_wrt_damage(d,d_old)*H*g) * ufl.dx
        
        #only for g = (1-d)**2, w=d
        # F = (g*3/8 + (3/4)*l**2*ufl.inner(ufl.grad(d), ufl.grad(g)) \
        #         - C3*l*2*(1-d)*H*g) * ufl.dx
        
        
        a, L = ufl.lhs(F), ufl.rhs(F)

        self.problem = fem.petsc.LinearProblem(a, L, self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        return self.problem.solve()
    
    def solve_nonlinear(self,v,Hprev,d):


        H = mf.history_function(ε(v),Hprev,self.material.ν,self.material.ψcritstar)

        C3 = self.material.C3; l = self.material.l

        free_energy = (mf.crack_density_function(d,l,self.w, self.c0) \
                       + C3*mf.degradation(d)*H)*ufl.dx

        F = ufl.derivative(free_energy,d,ufl.TestFunction(self.V))

        g = ufl.TestFunction(self.V)

        # only for g = (1-d)**2, w=d**2
        # F = (ufl.inner(d,g) + l**2*ufl.inner(ufl.grad(d), ufl.grad(g)) \
        #         - C3*l*2*(1-d)*H*g) * ufl.dx
        
        self.problem = fem.petsc.NonlinearProblem(F, d, self.bcs)
        
        self.solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-14
        self.solver.atol = 1e-14 
        self.solver.max_it = 50
        # self.solver.report = True

        

        ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
        opts[f"{option_prefix}pc_type"] = "lu"

        ksp.setFromOptions()

        n, converged = self.solver.solve(d)

        # d.x.array[:][d.x.array[:] > 1.0] = 1.0


    
class HistorySolver:
    def __init__(self, msh, material, dt=0.0):
        self.msh = msh
        self.material = material
        self.dt = dt
        

        h_el = bufl.element("DG", self.msh.basix_cell(), 0, dtype=default_real_type)
        self.V = fem.functionspace(self.msh, h_el)
        
    def solve(self, Hprev, v):


        h, g = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        H = mf.history_function(ε(v),Hprev,self.material.ν,self.material.ψcritstar)

        a = ufl.inner(h,g) * ufl.dx
        L = ufl.inner(H,g) * ufl.dx

        self.problem = fem.petsc.LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        return self.problem.solve()
    

    def advect(self, Hprev, u, k=1e-4):

        f = ufl.TrialFunction(self.V)
        g = ufl.TestFunction(self.V)
        
        n = ufl.FacetNormal(self.msh)
        h = ufl.CellDiameter(self.msh)

        α = 3.0

        Dt = lambda f: advection_numerics.backward_euler(f, Hprev, self.dt)
        a_A = lambda f, g: advection_numerics.advection(f, g, u, n)
        a_D = lambda f, g: advection_numerics.diffusion(f, g, k, α, n, h)

        F = Dt(f)*g*ufl.dx + a_A(f,g)+ a_D(f,g)

        a, L = ufl.lhs(F), ufl.rhs(F)

        problem = fem.petsc.LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        return problem.solve()


class BoundedSolver:
    def __init__(self, msh, bc_func, material):
        self.msh = msh
        self.material = material

        d_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        self.V = fem.functionspace(self.msh, d_el)

        self.bcs = bc_func(self.V)

        
        self.d_lb = fem.Function(self.V)
        self.d_ub = fem.Function(self.V)

        self.d_lb.x.array[:] = 0.0
        self.d_ub.x.array[:] = 1.0

    def setup_solver(self, v, d):
        self.v = v
        self.d = d
        
        energy = (mf.degraded_free_energy(ε(self.v),self.d,self.material.ν,self.material.ψcritstar)\
                    + (1/self.material.C3)*mf.crack_density_function(self.d,self.material.l,2))*ufl.dx
            
        E_d = ufl.derivative(energy,self.d,ufl.TestFunction(self.V))
        E_d_d = ufl.derivative(E_d,self.d,ufl.TrialFunction(self.V))

        self.problem = solvers.NonlinearPDE_SNESProblem(
            fem.form(E_d), fem.form(E_d_d), self.d, bcs=self.bcs)
        
        self.solver_d_snes = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver_d_snes.setType("vinewtonrsls")
        self.solver_d_snes.setFunction(self.problem.F_mono, fem.petsc.create_vector(fem.form(E_d)))
        self.solver_d_snes.setJacobian(self.problem.J_mono, fem.petsc.create_matrix(fem.form(E_d_d)),P=None)
        self.solver_d_snes.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver_d_snes.getKSP().setType("preonly")
        self.solver_d_snes.getKSP().setTolerances(rtol=1.0e-9)
        self.solver_d_snes.getKSP().getPC().setType("lu")
        # We set the bound (Note: they are passed as reference and not as values)
        self.solver_d_snes.setVariableBounds(self.d_lb.x.petsc_vec,self.d_ub.x.petsc_vec)

    def solve(self,v):

        self.v = v
        self.solver_d_snes.solve(None, self.d.x.petsc_vec)

        return self.d





            







        

    

