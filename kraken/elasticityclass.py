from dolfinx import fem, default_scalar_type, la, default_real_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
import phasefield as pf
from phasefield import ε
import basix.ufl as bufl
import nonlinear
from kraken import bodyforces as bf


class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, material, dt):
        self.msh = msh
        self.material = material
        self.dt = dt
        self.pw = bf.water_pressure

        h_el = bufl.element("DG", self.msh.basix_cell(), 0, dtype=default_real_type)
        self.Q_h = fem.functionspace(self.msh, h_el)

        
        v_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        
        self.V_e = fem.functionspace(self.msh, v_el)

        d_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)

        self.V_d = fem.functionspace(self.msh, d_el)

        P2_el = bufl.element("Lagrange", self.msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
        P1_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)

        self.P2 = fem.functionspace(self.msh, P2_el)
        self.P1 = fem.functionspace(self.msh, P1_el)

        TH = bufl.mixed_element([P2_el, P1_el])
        self.W = fem.functionspace(self.msh, TH)

        self.bcs_v = bc_funcs[0](self.V_e)
        self.bcs_u = bc_funcs[1](self.P2)
        self.bcs_d = bc_funcs[2](self.V_d)

        self.Hprev = 0.0


        self.v = fem.Function(self.V_e, name="elastic displacement")
        self.d = fem.Function(self.V_d, name="damage")
        self.u = fem.Function(self.P2, name="velocity")
        self.p = fem.Function(self.P1, name="pressure")

        self.x = fem.Function(self.W)

        # self.init_damage_limits()
    
    
    def η(self, u):
        n = self.material.n
        eps = 1.e-8
        return (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))


    def solve_elastic(self):

        C1 = self.material.C1; ν = self.material.ν
        ψcrit = self.material.ψcritstar; ρratio = self.material.ρratio

        ds = ufl.Measure("ds", domain=self.msh)

        n = ufl.FacetNormal(self.msh)

        pw = lambda v: self.pw(self.msh,v)# + self.dt*self.u)
        # pw = lambda v: -1.0
        g = lambda d: pf.degradation(d)

        f = bf.body_force(self.msh, ρratio)
        
        internal_energy = pf.degraded_free_energy(ε(self.v),self.d,ν,ψcrit) * ufl.dx

        ### for sneddon only
        # internal_energy = pf.degradation(self.d)*pf.free_energy(ε(self.v),ν) * ufl.dx

        external_energy =  C1 *( g(self.d)*ufl.dot(f, self.v) - pw(self.v)*ufl.inner(ufl.grad(g(self.d)), self.v) )* ufl.dx \
            - C1 * g(self.d) * pw(self.v) *  ufl.dot(n, self.v) * ds
        

        total_energy = internal_energy - external_energy

        self.F = ufl.derivative(total_energy,self.v,ufl.TestFunction(self.V_e))

        self.elastic_problem = NonlinearProblem(self.F, self.v, self.bcs_v)

        self.elastic_solver = Newton(self.elastic_problem)


        n, converged = self.elastic_solver.solve(self.v)
        assert(converged)


    def solve_damage(self):
        H = pf.history_function(ε(self.v),self.Hprev,self.material.ν,self.material.ψcritstar)
        # H = self.solve_history()

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

        
        pw = lambda u: self.pw(self.msh,u*self.dt + self.v)
        
        n = ufl.FacetNormal(self.msh)
        ds = ufl.Measure("ds", domain=self.msh)

        f = bf.body_force(self.msh, ρratio)
        
        p_v = -((self.material.λ/self.material.μ)+(2/self.msh.geometry.dim))*ufl.div(self.v)

        F = [((1/C2)*g*self.η(self.u)*ufl.inner(pf.ε(self.u), pf.ε(v)) \
        # + ufl.inner(hat(-self.p), ufl.div(v))\
        - ufl.inner(self.p, ufl.div(v)) \
        # + (g-1)*ufl.inner(pf.positive_part(-p_v), ufl.div(v)) \
        - C1 * g * ufl.inner(f, v) \
        + C1 * pw(self.u) * ufl.inner(ufl.grad(g), v)) * ufl.dx \
        + C1 * g * pw(self.u) * ufl.inner(n, v) * ds,
        # )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]

        J = get_jacobian(F,self.u,self.p,du,dp)
        P = get_preconditioner(J, self.u, dp, q, lambda u: g*self.η(u))

        snes, x = nonlinear.nested_solve(F, J, self.u, self.p, self.bcs_u, P)
        # snes, x = nonlinear.block_solve(F, J, P, self.u, self.p, self.bcs_u, self.P2, self.P1)

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()


    
    def solve_stokes_linearised(self):

        u, p = ufl.TrialFunction(self.P2), ufl.TrialFunction(self.P1)
        v, q = ufl.TestFunction(self.P2), ufl.TestFunction(self.P1)

        C1 = self.material.C1; C2 = self.material.C2
        ρratio = self.material.ρratio

        g = pf.degradation(self.d)

        pw = self.pw(self.msh,self.u*self.dt + self.v)

       
        η = self.η(self.u)

        n = ufl.FacetNormal(self.msh)
        ds = ufl.Measure("ds", domain=self.msh)

        f = bf.body_force(self.msh, ρratio)

        # p_v = -((self.material.λ/self.material.μ)+(2/self.msh.geometry.dim))*ufl.div(self.v)

        # F = [((1/C2)*g*self.η(self.u)*ufl.inner(pf.ε(u), pf.ε(v)) \
        #         # + ufl.inner(hat(-self.p), ufl.div(v))\
        #         - ufl.inner(self.p, ufl.div(v)) \
        #         # + (g-1)*ufl.inner(pf.positive_part(-p_v), ufl.div(v)) \
        #         - C1 * g * ufl.inner(f, v) \
        #         + C1 * pw(self.u) * ufl.inner(ufl.grad(g), v)) * ufl.dx \
        #         + C1 * g * pw(self.u) * ufl.inner(n, v) * ds,
        #         # )*ufl.dx,
        #         - ufl.inner(ufl.div(u), q) * ufl.dx ]
        
        a = fem.form([[(1/C2)*g*η*ufl.inner(pf.ε(u), pf.ε(v)) * ufl.dx,
                    #    + C1*δpw(u)*ufl.inner(n,v)*ds,
                        ufl.inner(p, ufl.div(v))*ufl.dx],

                    [ufl.inner(ufl.div(u), q) * ufl.dx,
                        None]])
        
        L = fem.form([(C1*g*ufl.inner(f,v) \
                     - C1*g*pw*ufl.inner(ufl.grad(g),v))*ufl.dx \
                    #  - (g-1)*ufl.inner(pf.positive_part(-self.p), ufl.div(v))*ufl.dx\
                     - C1*g*pw*ufl.inner(n,v)*ds,
                    ufl.inner(fem.Constant(self.msh, default_scalar_type(0.0)),q)*ufl.dx])
        
        P11 = fem.form((2*g*η)**-1 * p*q  * ufl.dx)
        P = [[a[0][0], None],
            [None, P11]]
        
        ksp, x, b = nonlinear.linear_nested_solver(a, L, self.u, self.p, self.bcs_u, P)

        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
        # nonlinear.linear_block_solver(a, L, P, self.u, self.p, self.bcs_u, self.P2, self.P1)




    
    def fixed_point(self, max_its=100, tol=1e-4, solve_stokes=False):
        L2_old = 0.0



        for i in range(max_its):

            self.solve_damage()
            self.solve_elastic()
            if solve_stokes:
                self.solve_stokes_linearised()
            # self.solver_d.solve(None, self.d.x.petsc_vec)
            # self.solve_damage_limits()
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")
            
            if error_L2 < tol:
                
                break

            L2_old = L2

        # Update history function as finished fixed point iteration
        self.Hprev = self.solve_history()

    def solve_history(self):

        h, v = ufl.TrialFunction(self.Q_h), ufl.TestFunction(self.Q_h)

        H = pf.history_function(ε(self.v),self.Hprev,self.material.ν,self.material.ψcritstar)

        a = ufl.inner(h,v) * ufl.dx
        L = ufl.inner(H,v) * ufl.dx

        problem = LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        h = problem.solve()

        return h

    


    def update_mesh(self):
        V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        uhh = fem.Function(V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.dt*uhh.x.array.reshape((-1, self.msh.geometry.dim))



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



