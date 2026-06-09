import adios4dolfinx
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



class AT2(Damage):
    def __init__(self, sim):
        super().__init__(sim)


        self.d_el_mixed = bufl.mixed_element([self.d_el, self.d_el])

        self.W = fem.functionspace(self.sim.msh, self.d_el_mixed)
        self.w = fem.Function(self.W, name="mixed function")
        self.d, self.lap = ufl.split(self.w)
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.d_prev_time, self.lap_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.w_prev_it2 = fem.Function(self.W, name="mixed function 2 iterations previous")
        self.w_prev_it3 = fem.Function(self.W, name="mixed function 3 iterations previous")
        self.d_prev_it, self.lap_prev_it = ufl.split(self.w_prev_it)
        self.d_prev_it2, self.lap_prev_it2 = ufl.split(self.w_prev_it2)
        self.d_prev_it3, self.lap_prev_it3 = ufl.split(self.w_prev_it3)

        self.D, _ = self.W.sub(0).collapse()

        # self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        # self.Hprev = fem.Function(self.H_space, name="history")


        bc_func_mod = lambda V: self.sim.bc_funcs[1](V.sub(0))
        self.bc_d = bc_func_mod(self.W)


    def update_bcs(self,new_bcs):
        bc_func_mod = lambda V: new_bcs(V.sub(0))
        self.bc_d = bc_func_mod(self.W)
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)
        self.setup_solver()


    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        
        
        H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        self.F = -C3*2*l*(1-self.d)*v*H*ufl.dx \
                  +(1/2)*(2*self.d*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                ) * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * ufl.dx
        

    

        C_new = 1e-2/(self.sim.params.Gc*self.sim.params.τ)

        d_dot = (self.d - self.d_prev_time)/self.sim.params.dtstar

        # self.F += C_new*l*d_dot*v*ufl.dx
        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def interpolate_from_parent(self, parent):
        super().interpolate_from_parent(parent)

        self.w.sub(0).interpolate(parent.damage.w.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w.sub(1).interpolate(parent.damage.w.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        
        self.w_prev_it.sub(0).interpolate(parent.damage.w_prev_it.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it.sub(1).interpolate(parent.damage.w_prev_it.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        self.w_prev_it2.sub(0).interpolate(parent.damage.w_prev_it2.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it2.sub(1).interpolate(parent.damage.w_prev_it2.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        self.w_prev_it3.sub(0).interpolate(parent.damage.w_prev_it3.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it3.sub(1).interpolate(parent.damage.w_prev_it3.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        

        self.w_prev_time.sub(0).interpolate(parent.damage.w_prev_time.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_time.sub(1).interpolate(parent.damage.w_prev_time.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

    def timestep(self):
        super().timestep()
        self.w_prev_time.x.array[:] = self.w.x.array[:]



    def solve(self):
        self.w_prev_it3.x.array[:] = self.w_prev_it2.x.array[:]
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        
        assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"

    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]
