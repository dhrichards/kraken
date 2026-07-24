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



class LowerOrder(Damage):
    ''' Parent class for lower order damage models, such as AT1 and AT2'''
    def __init__(self, sim):
        super().__init__(sim)


        self.D = fem.functionspace(self.sim.msh, self.d_el)
        self.d = fem.Function(self.D, name="damage")
        self.w = self.d # for compatibility
        self.d_prev_time = fem.Function(self.D, name="damage_prev_time")
        self.w_prev_time = self.d_prev_time  # for compatibility

        self.d_prev_it = fem.Function(self.D, name="damage_prev_it")
        self.d_prev_it2 = fem.Function(self.D, name="damage_prev_it2")
        self.d_prev_it3 = fem.Function(self.D, name="damage_prev_it3")

        self.g = es.degradation(self.d)

        

        self.bc_d = self.sim.bc_funcs[1](self.D)
        

    def solve(self):
        self.d_prev_it3.x.array[:] = self.d_prev_it2.x.array[:]
        self.d_prev_it2.x.array[:] = self.d_prev_it.x.array[:]
        self.d_prev_it.x.array[:] = self.d.x.array[:]
        self.solver.solve(None, self.d.x.petsc_vec)

        self.d.x.scatter_forward()
        
        assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"


   



class AT2(LowerOrder):
    '''Lower oder AT2 phase-field damage model.
      Irreversibility is enforced using a history variable, which is updated at each timestep.'''
    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
    

        v = ufl.TestFunction(self.D)

        
        self.F = (ufl.inner(self.d,v) + l**2*ufl.inner(ufl.grad(self.d), ufl.grad(v)) \
                - C3*l*2*(1-self.d)*H*v) * ufl.dx
        

        self.J = ufl.derivative(self.F,self.d,ufl.TrialFunction(self.D))


        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)




class AT1(LowerOrder):
    '''Lower order AT1 phase-field damage model.
      Solved using a bounded solver,
      Irreversibility is enforced using d_prev_time as a lower bound, which is updated at each timestep.'''
    def __init__(self, sim):
        super().__init__(sim)

        self.d_prev_time = fem.Function(self.D, name="damage_prev_time")

    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        w = lambda d: d
        s = np.linspace(0,1,500)
        c0 = 4*np.trapezoid(np.sqrt(w(s)),s)
        

        H = ufl.max_value(self.sim.momentum.ψplus - ψcrit, 0)

        dissipated_energy = (1/C3) * es.crack_density_function(self.d, l, w, c0) * ufl.dx
        elastic_energy = self.g * H * ufl.dx

        total_energy = dissipated_energy + elastic_energy

        self.F = ufl.derivative(total_energy, self.d, ufl.TestFunction(self.D))
        self.J = ufl.derivative(self.F, self.d, ufl.TrialFunction(self.D))

        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)

    def setup_solver(self):
        d_ub = fem.Function(self.D, name="damage_ub")
        d_ub.x.array[:] = 1.0

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(self.J)), P=None)

        self.solver.setType("vinewtonrsls")
        self.solver.setVariableBounds(self.d_prev_time.x.petsc_vec, d_ub.x.petsc_vec)

        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("cg")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("jacobi")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")

    def timestep(self):
        # Update the history variable
        self.d_prev_time.x.array[:] = self.d.x.array[:]


    def write_checkpoint(self, filename, t=0):
        adios4dolfinx.write_function(filename, self.w, name = "w_damage",time = t)
        
