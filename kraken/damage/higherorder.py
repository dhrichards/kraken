from .base import Damage
import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type, nls
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics.maths_functions import ε
from kraken.numerics import solvers
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np



class HigherOrder(Damage):
    def __init__(self, sim, free_energy_plus=es.free_energy_plus_lo):
        super().__init__(sim, free_energy_plus=es.free_energy_plus_lo)


        self.d_el_mixed = bufl.mixed_element([self.d_el, self.d_el])

        self.W = fem.functionspace(self.sim.msh, self.d_el_mixed)
        self.w = fem.Function(self.W, name="mixed function")
        self.d, self.lap = ufl.split(self.w)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.d_prev_time, self.lap_prev_time = ufl.split(self.w_prev_time)

        self.D, _ = self.W.sub(0).collapse()

        self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        self.Hprev = fem.Function(self.H_space, name="history")

        self.g = es.degradation_default(self.d)

        bc_func_mod = lambda V: self.sim.bc_funcs[1](V.sub(0))
        self.bc_d = bc_func_mod(self.W)


    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        

        l0 = l/2
        c = 1-self.d

        H = es.history_function(self.sim.momentum.ε_e, self.Hprev,
                            self.sim.params.ν, self.sim.params.ψcritstar,
                            self.free_energy_plus)

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        self.F = (C3*4*l0*c*v*H + c*v - 2*l0**2*self.lap*v - l0**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                -1.0*v ) * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(c), ufl.grad(q))) * ufl.dx
                
        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)



    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)

