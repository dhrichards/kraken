import numpy as np
from dolfinx import fem, default_scalar_type
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.momentum.base import Momentum
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc


class Stokes(Momentum):
    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)



    




class Direct(Stokes):
    def __init__(self, sim):
        super().__init__(sim)

        self.mixed_el = bufl.mixed_element([self.u_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        self.u, self.p = ufl.split(self.w)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.bc_u = self.sim.bc_funcs[0](self.W)

        self.vel = (self.u - self.u_prev_time)/ self.sim.params.dt
        self.vel_prev_it = (self.u_prev_it - self.u_prev_time) / self.sim.params.dt

        self.pw = self.water_pressure(self.u)
        self.ε_e = mf.ε(self.u)

    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = self.sim.damage.g
        η = mf.viscosity(mf.ε(self.vel_prev_it), self.sim.params.n, 1.e-8)

        f = mf.body_force(self.sim.msh, self.sim.params.ρistar)

        self.F= (
                g*η*ufl.inner(mf.ε(self.vel), mf.ε(v))\
                + ufl.inner(-self.p, ufl.div(v))  \
            -    ufl.inner(f, v) \
             ) * ufl.dx \
             + self.pw * ufl.inner(n, v) * ufl.ds \
        
        
        self.F += (
                - g*ufl.inner(ufl.div(self.vel), q) \
                ) * ufl.dx 

        self.J = ufl.derivative(self.F, self.w, ufl.TrialFunction(self.W))

        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]

    
    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]

