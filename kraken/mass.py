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


class Mass:
    def __init__(self, sim):
        self.sim = sim
        

        self.P = fem.functionspace(self.sim.msh, ("DG", 1))

        self.ρ = fem.Function(self.P, name="Density")
        self.ρ.x.array[:] = 1.0
        self.ρ_prev_time = fem.Function(self.P, name="Density previous time")
        self.ρ_prev_time.x.array[:] = 1.0



    def setup(self):

        # ρ = ufl.TrialFunction(self.P)
        v = ufl.TestFunction(self.P)
        
        dt = self.sim.params.dtstar


        self.F = ((self.ρ-self.ρ_prev_time)/dt*v + self.ρ*ufl.div(self.sim.params.ucstar*self.sim.momentum.du)/dt*v)*ufl.dx

        # a = ufl.lhs(self.F)
        # L = ufl.rhs(self.F)
    

        # self.problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.J = ufl.derivative(self.F, self.ρ, ufl.TrialFunction(self.P))
        
    

        # self.temperature_problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.problem = solvers.SNESProblem(self.F, self.ρ, bcs=[])

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setFunction(self.problem.F, dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)
        
        self.solver.setType("newtonls")
        
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")
    

    def solve(self):
        # ρ = self.problem.solve()
        # self.ρ.x.array[:] = ρ.x.array[:]
        self.solver.solve(None, self.ρ.x.petsc_vec)

    def timestep(self):
        self.ρ_prev_time.x.array[:] = self.ρ.x.array[:]
    


