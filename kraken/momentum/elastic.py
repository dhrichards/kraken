import numpy as np
from dolfinx import fem, default_scalar_type, mesh
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.momentum.base import Momentum
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers

class Elasticity(Momentum):
    '''Elasticity class for solving the momentum equation in a linear elastic material.'''
    def __init__(self, sim):
        super().__init__(sim)

        self.W = fem.functionspace(self.sim.msh, ("Lagrange", 2, (self.sim.msh.geometry.dim, )))

        self.u = fem.Function(self.W, name="displacement")
        self.w = self.u
        self.ε_e = mf.ε(self.u)
        self.ψplus = self.free_energy_plus(self.ε_e,self.u)
        self.p_crack = self.crack_pressure(self.u)
        self.ψplus  = es.free_energy_plus_nt(self.ε_e + self.p_crack*ufl.Identity(self.sim.msh.geometry.dim)/(3*es.Koverμ(self.sim.params.ν)), self.sim.params.ν)

        self.u_prev_it = fem.Function(self.W, name="displacement previous iteration")
        self.u_prev_time = fem.Function(self.W, name="displacement previous time")

        self.bc_u = self.sim.bc_funcs[0](self.W)

        


    def setup_momentum(self):
        
        v = ufl.TestFunction(self.W)
          
        p_w = self.water_pressure(self.u)
 
        σ = self.stress(self.ε_e, self.u)

     
        
        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

       
        self.F = (ufl.inner(σ, mf.ε(v))\
                #   -self.p_crack*ufl.inner(ufl.grad(g),v)
              - ufl.inner(f, v) 
              ) * ufl.dx 
        
        self.F += (
              + p_w * ufl.inner(n, v) * ufl.ds 
        )
            
        self.J = ufl.derivative(self.F,self.u,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_u)

    def solve(self):
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u_prev_it.x.array[:] = self.u.x.array[:]

    def timestep(self):
        self.u_prev_time.x.array[:] = self.u.x.array[:]


