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



class WithGamma(Damage):
    def __init__(self, sim):
        super().__init__(sim)

        self.D = fem.functionspace(self.sim.msh, ("Lagrange", 1))
        self.Γ = fem.functionspace(self.sim.msh, ("Lagrange", 1))
        self.d = fem.Function(self.D, name="damage")
        self.γtr = fem.Function(self.Γ, name="gamma")
        self.γ_prev = fem.Function(self.Γ, name="gamma previous time")

        self.γ = ufl.max_value(self.γtr, self.γ_prev)

        self.g = 1/(1 + self.γ) # degradation function to modify the stress



        

        self.bc_d = self.sim.bc_funcs[1](self.D)


    def setup_weak_form(self):
        self.setup_gamma()
        self.setup_damage()

    def setup_solver(self):
        self.setup_gamma_solver()
        self.setup_damage_solver()
        

    def setup_gamma(self):
        
        ψc = self.sim.params.ψcritstar
        v = ufl.TestFunction(self.Γ)
        ψp = self.sim.free_energy_plus(self.sim.momentum.ε_e, 
                                   self.sim.params.ν)
        
        gq = es.degradation_default(self.d)

        self.Fγ = (ψp/(1+self.γtr)**2 - gq*ψc) * v * ufl.dx 

        self.Jγ = ufl.derivative(self.Fγ, self.γtr, ufl.TrialFunction(self.Γ))

        self.gamma_problem = solvers.SNESProblem(self.Fγ, self.γtr, bcs=[])


    def setup_damage(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        H = ψcrit * self.γtr
        v = ufl.TestFunction(self.D)

        
        self.Fd = (ufl.inner(self.d,v) + l**2*ufl.inner(ufl.grad(self.d), ufl.grad(v)) \
                - C3*l*2*(1-self.d)*H*v) * ufl.dx
        

        self.Jd = ufl.derivative(self.Fd,self.d,ufl.TrialFunction(self.D))


        self.damage_problem = solvers.SNESProblem(self.Fd, self.d, bcs=self.bc_d)

    def setup_gamma_solver(self):

        self.gamma_solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.gamma_solver.setFunction(self.gamma_problem.F, fem.petsc.create_vector(fem.form(self.Fγ)))
        self.gamma_solver.setJacobian(self.gamma_problem.J, fem.petsc.create_matrix(fem.form(self.Jγ)),P=None)
        
        self.gamma_solver.setType("newtonls")
        
        self.gamma_solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.gamma_solver.getKSP().setType("preonly")
        self.gamma_solver.getKSP().setTolerances(rtol=1.0e-9)
        self.gamma_solver.getKSP().getPC().setType("lu")

    def setup_damage_solver(self):

        self.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.damage_solver.setFunction(self.damage_problem.F, fem.petsc.create_vector(fem.form(self.Fd)))
        self.damage_solver.setJacobian(self.damage_problem.J, fem.petsc.create_matrix(fem.form(self.Jd)),P=None)
        
        self.damage_solver.setType("newtonls")
        
        self.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.damage_solver.getKSP().setType("preonly")
        self.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
        self.damage_solver.getKSP().getPC().setType("lu")


    def solve(self):
        self.gamma_solver.solve(None, self.γtr.x.petsc_vec)
        self.damage_solver.solve(None, self.d.x.petsc_vec)


    def timestep(self):
        self.γ_prev.interpolate(fem.Expression(self.γ, self.Γ.element.interpolation_points()))




