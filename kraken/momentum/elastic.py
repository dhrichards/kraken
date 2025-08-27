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


class Elasticity(Momentum):
    def __init__(self, sim):
        super().__init__(sim)

        self.U = fem.functionspace(self.sim.msh, ("Lagrange", 1, (self.sim.msh.geometry.dim, )))

        self.u = fem.Function(self.U, name="displacement")
        self.ε_e = mf.ε(self.u)

        self.u_prev_it = fem.Function(self.U, name="displacement previous iteration")
        self.u_prev_time = fem.Function(self.U, name="displacement previous time")

        self.bc_u = self.sim.bc_funcs[0](self.U)


    def setup_momentum(self):
        
        v = ufl.TestFunction(self.U)

        g = self.sim.damage.g

        p_w = mf.water_pressure(self.sim.msh,self.u,self.sim.params.ucstar) +self.sim.params.patmstar
        p_crack = mf.water_pressure(self.sim.msh, self.u, self.sim.params.ucstar, level=0.00) + self.sim.params.patmstar

        
        
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        σ = g*σplus + σminus
        # σ = self.g*σ0

        # σ = pt.degraded_stress(self.ε_e, mf.ε(self.u_prev_it), self.g, self.params.ν)

        
        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

        d = self.sim.damage.d
        # Iprime = 2 - 2*d # Iprime*grad(d) = -grad(g)
        Iprime = 2*self.sim.damage.d
        # Iprime = 1.0



        self.F = (ufl.inner(σ, mf.ε(v))\
              - ufl.inner(f, v) 
              -p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
            #  + p_crack* ufl.inner(Iprime*ufl.grad(d), v)\
              ) * ufl.dx \
            + p_w * ufl.inner(n, v) * ufl.ds 
        

        self.J = ufl.derivative(self.F,self.u,ufl.TrialFunction(self.U))
            
        
        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_u)

    def solve(self):
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u_prev_it.x.array[:] = self.u.x.array[:]

    def timestep(self):
        self.u_prev_time.x.array[:] = self.u.x.array[:]


