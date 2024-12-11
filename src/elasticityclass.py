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

def viscosity(u, n, eps=1.e-8, A=1.0): 
    return A**(-1/n) * (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))



class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, material):
        self.msh = msh
        self.material = material

        
        v_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        
        self.V_e = fem.functionspace(self.msh, v_el)

        d_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)

        self.V_d = fem.functionspace(self.msh, d_el)

        P2_el = bufl.element("Lagrange", self.msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
        P1_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)

        self.P2 = fem.functionspace(self.msh, P2_el)
        self.P1 = fem.functionspace(self.msh, P1_el)

        self.bcs_v = bc_funcs[0](self.V_e)
        self.bcs_u = bc_funcs[1](self.P2)
        self.bcs_d = bc_funcs[2](self.V_d)

        self.Hprev = 0.0


        self.v = fem.Function(self.V_e, name="elastic displacement")
        self.d = fem.Function(self.V_d, name="Damage")
        self.u = fem.Function(self.P2, name="velocity")
        self.p = fem.Function(self.P1, name="pressure")


    def solve_elastic(self):

        C1 = self.material.C1; ν = self.material.ν
        ψcrit = self.material.ψcritstar; ρratio = self.material.ρratio

        ds = ufl.Measure("ds", domain=self.msh)

        n = ufl.FacetNormal(self.msh)

        pw = lambda u: bf.water_pressure(self.msh,u)
        g = lambda d: pf.degradation(d)
        f = pf.body_force(self.msh, ρratio)
        
        internal_energy = pf.degraded_free_energy(ε(self.v),self.d,ν,ψcrit) * ufl.dx
        # internal_energy = (pf.degradation(d)*free_energy(u,ν) + (1/C3)*pf.γ(d,l)) * ufl.dx

        external_energy =  C1 *( g(self.d)*ufl.dot(f, self.v) - pw(self.v)*ufl.inner(ufl.grad(g(self.d)), self.v) )* ufl.dx \
            - C1 * g(self.d) * pw(self.v) *  ufl.dot(n, self.v) * ds
        

        total_energy = internal_energy - external_energy

        self.F = ufl.derivative(total_energy,self.v,ufl.TestFunction(self.V_e))

        self.problem = NonlinearProblem(self.F, self.v, self.bcs_v)

        self.solver = Newton(self.problem)


        n, converged = self.solver.solve(self.v)
        assert(converged)


    def solve_damage(self):
        H = pf.history_function(ε(self.v),self.material,self.Hprev)

        d = ufl.TrialFunction(self.V_d)
        v = ufl.TestFunction(self.V_d)

        C3 = self.material.C3; l = self.material.l

        F = (ufl.inner(d,v) + l**2*ufl.inner(ufl.grad(d), ufl.grad(v)) \
             - C3*l*2*(1-d)*H*v) * ufl.dx
        
        a, L = ufl.lhs(F), ufl.rhs(F)

        self.damage_problem = LinearProblem(a, L, self.bcs_d, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        
        self.d = self.damage_problem.solve()

    def solve_stokes(self):

        du, dp = ufl.TrialFunction(self.P2), ufl.TrialFunction(self.P1)
        v, q = ufl.TestFunction(self.P2), ufl.TestFunction(self.P1)

        C1 = self.material.C1; C2 = self.material.C2
        ρratio = self.material.ρratio

        g = pf.degradation(self.d)

        def η(u):
            return g*viscosity(u, self.material.n, 1.e-8)
        
        pw = lambda u: bf.water_pressure(self.msh,u*self.dt + self.v)
        
        n = ufl.FacetNormal(self.msh)
        ds = ufl.Measure("ds", domain=self.msh)

        f = bf.body_force(self.msh, ρratio)

        F = [((1/C2)*η(self.u)*ufl.inner(pf.ε(self.u), pf.ε(v)) \
        # + ufl.inner(hat(-p), ufl.div(v))\
        - ufl.inner(self.p, ufl.div(v)) \
        - C1 * g * ufl.inner(f, v) \
        + C1 * pw(self.u) * ufl.inner(ufl.grad(g), v)) * ufl.dx \
        + C1 * g * pw(self.u) * ufl.inner(n, v) * ds,
        # )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]

        J = get_jacobian(F,self.u,self.p,du,dp)
        P = get_preconditioner(J, self.u, dp, q, η)

        snes, x = nonlinear.nested_solve(F, J, self.u, self.p, self.bcs_u, P)

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
    
        


    
    def minimisation(self, max_its=100, tol=1e-4):
        L2_old = 0.0

        for i in range(max_its):

            
            self.solve_elastic()
            self.solve_damage()

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)
            print(f"iteration {i}, error {error_L2}")
            
            if error_L2 < tol:
                
                break

            L2_old = L2

        # Update history function as finished fixed point iteration
        self.Hprev = pf.history_function(ε(self.v),self.material,self.Hprev)


    def update_mesh(self, k=1.0):
        V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        uhh = fem.Function(V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += k*uhh.x.array.reshape((-1, self.msh.geometry.dim))






    






def Newton(problem):

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1.0e-8
    solver.atol = 1.0e-8
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

    return solver

def get_jacobian(F,u,p,du,dp):
    return [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
            [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)]]

def get_preconditioner(J, u, dp, q, η):
    return [[J[0][0], None],
            [None, (2 * η(u))**-1 * dp * q * ufl.dx]]



