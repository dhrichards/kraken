from dolfinx import fem, default_real_type, default_scalar_type
import basix.ufl as bufl
import ufl
from kraken.numerics import maths_functions as mf
from kraken.numerics.maths_functions import ε
from kraken.numerics import solvers
from petsc4py import PETSc

class StokesSolver:
    def __init__(self, msh, bc_func, material, dt):
        self.msh = msh
        self.material = material
        self.dt = dt

        P2_el = bufl.element("Lagrange", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        P1_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)

        self.V = fem.functionspace(self.msh, P2_el)
        self.Q = fem.functionspace(self.msh, P1_el)

        self.bcs = bc_func(self.V)



    def setup(self,u,p,d,v):
        self.u = u
        self.p = p
        self.d = d
        self.v = v

        du, dp = ufl.TrialFunction(self.V), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.V), ufl.TestFunction(self.Q)

        C1 = self.material.C1; C2 = self.material.C2
        ρratio = self.material.ρratio


        # Phase field changes
        g = mf.degradation(self.d)

        def η(u):
            return mf.viscosity(u, self.material.n, 1.e-8)
        
        
        f = mf.body_force(self.msh, ρratio, self.material.slope_angle)


        n = ufl.FacetNormal(self.msh)           
        ds = ufl.Measure("ds", domain=self.msh)

        # Water pressure
        pw = lambda u : mf.water_pressure(self.msh,self.v + u*self.dt)
        

        
        
        F = [((1/C2)*g*η(self.u)*ufl.inner(ε(self.u), ε(v)) \
        - ufl.inner(self.p, ufl.div(v)) \
        - C1 * g * ufl.inner(f, v) \
        + C1 * pw(self.u) * ufl.inner(ufl.grad(g), v)) * ufl.dx \
        + C1 * g * pw(self.u) * ufl.inner(n, v) * ds,
        # )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]
        
        J = [[ufl.derivative(F[0], self.u, du), ufl.derivative(F[0], self.p, dp)],
            [ufl.derivative(F[1], self.u, du), ufl.derivative(F[1], self.p, dp)]]
        
        P = [[J[0][0], None],
            [None, (2 * g*η(self.u))**-1 * dp * q * ufl.dx]]

        self.solver, self.x = solvers.nested_solve(F, J, u, p, self.bcs, P)

        opts = PETSc.Options()
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"
        
        # opts["snes_rtol"] = 1.0e-7
        self.solver.setFromOptions()


    def solve(self):

        self.solver.solve(None, self.x)
        assert self.solver.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()

    def solve_semi_linearised(self,u,p,d,v):

        self.u = u
        self.p = p
        self.d = d
        self.v = v

        du, dp = ufl.TrialFunction(self.V), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.V), ufl.TestFunction(self.Q)

        C1 = self.material.C1; C2 = self.material.C2
        ρratio = self.material.ρratio


        # Phase field changes
        g = mf.degradation(self.d)

        def η(u):
            return mf.viscosity(u, self.material.n, 1.e-8)
        
        
        f = mf.body_force(self.msh, ρratio, self.material.slope_angle)


        n = ufl.FacetNormal(self.msh)           
        ds = ufl.Measure("ds", domain=self.msh)

        # Water presure
        pw = lambda u : mf.water_pressure(self.msh,self.v + u*self.dt)
        

        η = mf.viscosity(u, self.material.n)

        F = [((1/C2)*g*η*ufl.inner(ε(self.u), ε(v)) \
        - ufl.inner(self.p, ufl.div(v)) \
        - C1 * g * ufl.inner(f, v) \
        + C1 * pw(self.u) * ufl.inner(ufl.grad(g), v)) * ufl.dx \
        + C1 * g * pw(self.u) * ufl.inner(n, v) * ds,
        # )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]
        
        J = [[ufl.derivative(F[0], self.u, du), ufl.derivative(F[0], self.p, dp)],
            [ufl.derivative(F[1], self.u, du), ufl.derivative(F[1], self.p, dp)]]
        
        P = [[J[0][0], None],
            [None, (2 * g*η(self.u))**-1 * dp * q * ufl.dx]]

        self.solver, self.x = solvers.nested_solve(F, J, u, p, self.bcs, P)

        self.solver.solve(None, self.x)
        assert self.solver.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()

        


    def solve_linearised(self,u_prev,p_prev,d,v):
        self.u = u_prev
        self.p = p_prev


        u, p = ufl.TrialFunction(self.V), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.V), ufl.TestFunction(self.Q)

        C1 = self.material.C1; C2 = self.material.C2
        ρratio = self.material.ρratio

        g = mf.degradation(d)

        pw = mf.water_pressure(self.msh,self.u*self.dt + v)

       
        η = mf.viscosity(self.u, self.material.n)

        n = ufl.FacetNormal(self.msh)
        ds = ufl.Measure("ds", domain=self.msh)

        f = mf.body_force(self.msh, ρratio, self.material.slope_angle)

        a = fem.form([[(1/C2)*g*η*ufl.inner(mf.ε(u), mf.ε(v)) * ufl.dx,
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
        
        ksp, x, b = solvers.linear_nested_solver(a, L, self.u, self.p, self.bcs_u, P)

        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
        # nonlinear.linear_block_solver(a, L, P, self.u, self.p, self.bcs_u, self.P2, self.P1)



    


