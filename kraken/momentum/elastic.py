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

        g = es.degradation_default(self.sim.damage.d,1e-12)

        p_w = self.water_pressure(self.u)
        p_crack = self.crack_pressure(self.u)

        
        
        σ = self.stress(self.ε_e)
    
        
        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

        d = self.sim.damage.d
        # Iprime = 2 - 2*d # Iprime*grad(d) = -grad(g)
        Iprime = 2*self.sim.damage.d
        # Iprime = 1.0



        self.F = (ufl.inner(σ, mf.ε(v))\
              - ufl.inner(f, v) 
              -p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
            # - p_crack*ufl.inner(ufl.grad(g), v) \
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


class ElasticDegraded(Elasticity):
    def setup_momentum(self):
        

        v = ufl.TestFunction(self.U)

        g = self.sim.damage.g

        p_w = self.water_pressure(self.u)
        p_crack = self.crack_pressure(self.u)

        
        
        σ = self.stress(self.ε_e)
    
        
        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

        d = self.sim.damage.d
        # Iprime = 2 - 2*d # Iprime*grad(d) = -grad(g)
        Iprime = 2*self.sim.damage.d
        # Iprime = 1.0



        self.F = (ufl.inner(σ, mf.ε(v))\
              - g*ufl.inner(f, v) 
            #   -p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
            - p_crack*ufl.inner(ufl.grad(g), v) \
            #  + p_crack* ufl.inner(Iprime*ufl.grad(d), v)\
              ) * ufl.dx \
            + p_w * ufl.inner(n, v) * ufl.ds 
        

        self.J = ufl.derivative(self.F,self.u,ufl.TrialFunction(self.U))
            
        
        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_u)



class ElasticEnergySplit(Elasticity):


    def setup_momentum(self):
        g = es.degradation_default(self.sim.damage.d,1e-12)

        p_w = self.water_pressure(self.u)
        p_crack = self.crack_pressure(self.u)

        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

        ψ0 = es.free_energy(self.ε_e, self.sim.params.ν)
        ψplus = self.sim.free_energy_plus(self.ε_e, self.sim.params.ν)
        ψminus = ψ0 - ψplus

        energy = (
                g*ψplus + ψminus - ufl.inner(f,self.u) \
                - p_crack*ufl.inner(ufl.Dx(g, 0), self.u[0]) \
                    )*ufl.dx \
                + p_w * ufl.inner(n, self.u) * ufl.ds
        
        self.F = ufl.derivative(energy, self.u, ufl.TestFunction(self.U))
        self.J = ufl.derivative(self.F,self.u,ufl.TrialFunction(self.U))

        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_u)


class ElasticPressure(Momentum):

    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        self.u, self.p = ufl.split(self.w)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.bc_u = self.sim.bc_funcs[0](self.W)

        
        self.pw = self.water_pressure(self.u)
        self.ε_e = mf.ε(self.u)


    def setup_momentum(self):
        
        w_test = ufl.TestFunction(self.W)
        v, q = ufl.split(w_test)

        g = es.degradation_default(self.sim.damage.d,1e-12)

        p_w = self.water_pressure(self.u)
        p_crack = self.crack_pressure(self.u)

        
        
        # σ = self.stress(self.ε_e)
        σ0 = -self.p*ufl.Identity(self.sim.msh.geometry.dim) + 2*(self.ε_e)

        # σplus = es.stress_plus_lo_pressure(self.ε_e, self.p,self.sim.params.ν)
        σplus = es.stress_plus_dp_pressure(self.ε_e, self.p,self.sim.params.ν)
        σminus = σ0 - σplus

        σ = g*σplus + σminus
    
        
        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

        d = self.sim.damage.d
        # Iprime = 2 - 2*d # Iprime*grad(d) = -grad(g)
        Iprime = 2*self.sim.damage.d
        # Iprime = 1.0


        λ = es.λoverμ(self.sim.params.ν)


        self.F = (ufl.inner(σ, mf.ε(v))\
              - ufl.inner(f, v) 
              -p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
            # - p_crack*ufl.inner(ufl.grad(g), v) \
            #  + p_crack* ufl.inner(Iprime*ufl.grad(d), v)\
              ) * ufl.dx \
            + p_w * ufl.inner(n, v) * ufl.ds 
        
        self.F += (
            + λ*ufl.inner(ufl.div(self.u), q)
            + self.p*q
             ) * ufl.dx
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)

    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w_prev_it.x.array[:] = self.w.x.array[:]

    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]

            




        

