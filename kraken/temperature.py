import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import energy_splits_deviatoric as esd
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc
import adios4dolfinx


class Temperature:
    def __init__(self, sim):
        self.sim = sim
        

        self.T_space = fem.functionspace(self.sim.msh, ("CG", 1))

        self.T = fem.Function(self.T_space, name="Temperature")
        self.T_prev = fem.Function(self.T_space, name="Temperature previous time")
        self.T_prev.x.array[:] = 273.15 + self.sim.params.T.value



    def setup(self):

        v = ufl.TestFunction(self.T_space)
        

        κ = self.sim.params.κstar
        C = self.sim.params.C_temperature
        dt = self.sim.params.dtstar
        A = mf.rate_factor(self.T_prev)/self.sim.params.A0
        g = es.degradation_default(self.sim.damage.d,self.sim.params.ge_tol)

        self.F = (self.T - self.T_prev)/dt*v*ufl.dx + κ*ufl.inner(ufl.grad(self.T), ufl.grad(v))*ufl.dx \
            - C*g*mf.viscous_energy(ufl.dev(mf.ε(self.sim.momentum.vel)), self.sim.params.n, A=A)*v*ufl.dx
        
        self.J = ufl.derivative(self.F, self.T, ufl.TrialFunction(self.T_space))
        
    

        # self.temperature_problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.problem = solvers.SNESProblem(self.F, self.T, bcs=[])

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setFunction(self.problem.F, dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)
        
        self.solver.setType("newtonls")
        
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")


    def solve(self):
        self.solver.solve(None, self.T.x.petsc_vec)

    def timestep(self):
        self.T_prev.x.array[:] = self.T.x.array[:]
    


