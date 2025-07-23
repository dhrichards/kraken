import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import numpy as np
from kraken.models import damage
from kraken.numerics import maths_functions as mf
from kraken.numerics import total_velocity_maths as mt
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc

class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, params,):
        self.msh = msh
        self.params = params

        
        self.U = fem.functionspace(self.msh, ("Lagrange", 2, (self.msh.geometry.dim,)))
        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim,)))
        self.Q = fem.functionspace(self.msh, ("Lagrange", 1))
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))
        self.H_space = fem.functionspace(self.msh, ("DG", 1))

        self.bc_u = bc_funcs[0](self.U)
        self.bc_d = bc_funcs[1](self.D)
        
        self.u = fem.Function(self.U, name="velocity")
        self.dp = fem.Function(self.Q, name="pressure")
        self.u_prev_it = fem.Function(self.U, name="velocity_prev")
        self.u_prev_it.x.array[:] = 1.0
        self.dp_prev_it = fem.Function(self.Q, name="pressure_prev_it")
        self.p_prev_time = fem.Function(self.Q, name="pressure_prev")
        
        self.p = self.p_prev_time + self.dp
        


        self.d = fem.Function(self.D, name="damage")
        self.d_prev_time = fem.Function(self.D, name="damage_prev_time")
        self.Hprev = fem.Function(self.H_space, name="history")

        self.g = mf.degradation_default(self.d)
        
        

    def update_history(self):

        H = mf.history_function(self.ε_e,self.Hprev,
                                self.params.ν,self.params.ψcritstar)

        self.Hprev.interpolate(fem.Expression(H,self.H_space.element.interpolation_points()))


    def setup_all(self,compressible=True):
        self.setup(compressible)
        damage.setup_damage_bounded(self)

    def setup(self,compressible=True):

        δt = self.params.dtstar
        λoverμ = self.params.λ/self.params.μ
        D = self.msh.geometry.dim
        κ = λoverμ + 2/D

        du, dp = ufl.TrialFunction(self.U), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.U), ufl.TestFunction(self.Q)

        n = ufl.FacetNormal(self.msh)
        
        p_ext = mf.water_pressure(self.msh,self.u,self.params.ucstar*self.params.dtstar) +self.params.patmstar
        # p_ext = mf.water_pressure_static(self.msh) + self.params.patmstar
        f = self.g*mf.body_force(self.msh, self.params.ρistar, self.params.slope_angle)
        
        self.η = mf.viscosity(mf.εD(self.u_prev_it), self.params.n)

        
        dotp = self.dp/δt
        dotp_prev_it = self.dp_prev_it/δt
        self.ε_e = self.η*mf.εD(self.u) - self.p/(D*self.g*κ) * ufl.Identity(D)


        F = [(self.g*2*self.η*ufl.inner(mf.ε(self.u), mf.ε(v)) \
        - ufl.inner(self.p \
                    # + 2*self.η*dot_p/(D*κ)\
                        , ufl.div(v)) \
        - ufl.inner(f, v) \
        - p_ext * ufl.inner(ufl.grad(self.g), v)\
            ) * ufl.dx \
        + self.g * p_ext* ufl.inner(n, v) * ufl.ds,
        - (ufl.inner(ufl.div(self.u), q) \
        #    +dotp*q/(self.g*κ)\
              )* ufl.dx ]
        
        J = [[ufl.derivative(F[0], self.u, du), ufl.derivative(F[0], self.dp, dp)],
            [ufl.derivative(F[1], self.u, du), ufl.derivative(F[1], self.dp, dp)]]
        
        P = [[J[0][0], None],
            [None, (2 * self.g*self.η)**-1 * dp * q * ufl.dx]]
        

        self.stokes_solver, self.x = solvers.nested_solve(F, J, self.u, self.dp, self.bc_u, P)

        opts = PETSc.Options()
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"
        
        # opts["snes_rtol"] = 1.0e-7
        self.stokes_solver.setFromOptions()

        

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def solve(self):
        # self.stokes.solve(self.u, self.p, self.d, self.v)
        self.stokes_solver.solve(None, self.x)

        self.u.x.scatter_forward()
        self.dp.x.scatter_forward()

        self.u_prev_it.x.array[:] = self.u.x.array[:]
        self.dp_prev_it.x.array[:] = self.dp.x.array[:]
        
  
    
    def timestep(self):


        
        uhh = fem.Function(self.V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.ucstar*self.params.dtstar*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        
        self.d_prev_time.x.array[:] = self.d.x.array[:]
        self.p_prev_time.x.array[:] += self.dp.x.array[:]

