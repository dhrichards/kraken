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
    def __init__(self, sim):
        super().__init__(sim)


        self.D = fem.functionspace(self.sim.msh, self.d_el)
        self.d = fem.Function(self.D, name="damage")
        self.w = self.d # for compatibility
        self.d_prev_time = fem.Function(self.D, name="damage_prev_time")
        self.w_prev_time = self.d_prev_time  # for compatibility

        self.g = es.degradation_default(self.d)

        

        self.bc_d = self.sim.bc_funcs[1](self.D)
        

    def solve(self):
        self.solver.solve(None, self.d.x.petsc_vec)

   



class NonLinear(LowerOrder):
    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
    

        v = ufl.TestFunction(self.D)

        
        self.F = (ufl.inner(self.d,v) + l**2*ufl.inner(ufl.grad(self.d), ufl.grad(v)) \
                - C3*l*2*(1-self.d)*H*v) * ufl.dx
        

        self.J = ufl.derivative(self.F,self.d,ufl.TrialFunction(self.D))


        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)


class NonLinearAT1(LowerOrder):
    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        H = es.history_function(self.sim.momentum.ε_e, self.Hprev,
                            self.sim.params.ν, self.sim.params.ψcritstar, 
                            self.sim.free_energy_plus)
    
        HH = ufl.max_value(C3*8*H/3 - 0.5*l, 0)
        v = ufl.TestFunction(self.D)

        
        self.F = (ufl.inner(self.d,v) + 2*l**2*ufl.inner(ufl.grad(self.d), ufl.grad(v)) \
                - l*2*(1-self.d)*HH*v) * ufl.dx
        

        self.J = ufl.derivative(self.F,self.d,ufl.TrialFunction(self.D))


        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)




class Bounded(LowerOrder):
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
        


class PressurisedCrack(Bounded):
    def setup_weak_form(self):
        super().setup_weak_form()

 
        # pw = mf.water_pressure(self.sim.msh, self.sim.momentum.u, self.sim.params.ucstar) + self.sim.params.patmstar
        pw = mf.water_pressure_static(self.sim.msh)
        # Iprime = 2 - 2*model.d
        Iprime = 2*self.d

        pressure_work = pw*ufl.inner(Iprime*ufl.Dx(self.d,0), self.sim.momentum.u[0]) * ufl.dx

        self.F += ufl.derivative(pressure_work, self.d, ufl.TestFunction(self.D))

        self.J = ufl.derivative(self.F, self.d, ufl.TrialFunction(self.D))

        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)


class Anisotropic(Bounded):
    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        w = lambda d: d
        s = np.linspace(0,1,500)
        c0 = 4*np.trapezoid(np.sqrt(w(s)),s)
        

        H = ufl.max_value(self.sim.free_energy_plus(self.sim.momentum.ε_e, ν) - ψcrit, 0)

        n = ufl.Constant(self.sim.msh, (0.0, 1.0))
        N = ufl.outer(n,n)
        A = ufl.Identity(2) + 5*N

        γ = (w(self.d)/l + l * ufl.inner(ufl.grad(self.d), ufl.grad(self.d)))/c0

        dissipated_energy = (1/C3) * es.crack_density_function(self.d, l, w, c0) * ufl.dx
        elastic_energy = self.g * H * ufl.dx

        total_energy = dissipated_energy + elastic_energy

        self.F = ufl.derivative(total_energy, self.d, ufl.TestFunction(self.D))
        self.J = ufl.derivative(self.F, self.d, ufl.TrialFunction(self.D))

        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)




