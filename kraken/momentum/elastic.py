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
    def __init__(self, sim):
        super().__init__(sim)

        self.W = fem.functionspace(self.sim.msh, ("Lagrange", 2, (self.sim.msh.geometry.dim, )))

        self.u = fem.Function(self.W, name="displacement")
        self.w = self.u
        self.ε_e = mf.ε(self.u)
        # self.ψplus = self.sim.free_energy_plus(self.ε_e, self.sim.params.ν)
        self.p_crack = self.crack_pressure(self.u)
        self.ψplus  = es.free_energy_plus_lo(self.ε_e + self.p_crack*ufl.Identity(self.sim.msh.geometry.dim)/(3*es.Koverμ(self.sim.params.ν)), self.sim.params.ν)

        self.u_prev_it = fem.Function(self.W, name="displacement previous iteration")
        self.u_prev_time = fem.Function(self.W, name="displacement previous time")

        self.bc_u = self.sim.bc_funcs[0](self.W)

        


    def setup_momentum(self):
        
        v = ufl.TestFunction(self.W)

        g = es.degradation_default(self.sim.damage.d,1e-12)

        p_w = self.water_pressure(self.u)
        p_crack = self.crack_pressure(self.u)

        # Efactor = fem.Function(fem.functionspace(self.sim.msh, ("CG", 1)))
        # Efactor.x.array[:] = 1
        # y = self.sim.msh.geometry.x[:,1]
        # Efactor.x.array[y>10] = 1e4
        
        σ = self.stress(self.ε_e)

        # σplus, F = fs.stress_plus(self.ε_e, self.sim.params.ν)

        # σ0 = es.λoverμ(self.sim.params.ν)*ufl.tr(self.ε_e)*ufl.Identity(2) + 2*self.ε_e
        # σminus = σ0 - σplus
        # σ = g*σplus + σminus

        # self.ψplus = F
    
        
        f = self.sim.params.ρistar*mf.body_force(self.sim.msh)

        n = ufl.FacetNormal(self.sim.msh)

        d = self.sim.damage.d
        # Iprime = 2 - 2*d # Iprime*grad(d) = -grad(g)
        Iprime = 2*self.sim.damage.d
        # Iprime = 1.0

        def right_boundary(x):
            return np.isclose(x[0], self.sim.params.length.value/self.sim.params.H.value/2)
        
        def bottom_boundary(x):
            return np.isclose(x[1], 0.0)
        
        def left_boundary(x):
            return np.isclose(x[0], 0.0)

        r_facets = mesh.locate_entities_boundary(self.sim.msh, self.sim.msh.topology.dim-1, right_boundary)
        b_facets = mesh.locate_entities_boundary(self.sim.msh, self.sim.msh.topology.dim-1, bottom_boundary)
        l_facets = mesh.locate_entities_boundary(self.sim.msh, self.sim.msh.topology.dim-1, left_boundary)
        facets = np.hstack([r_facets, b_facets, l_facets])
        values = np.hstack([np.full_like(r_facets, 1), np.full_like(b_facets, 2), np.full_like(l_facets, 3)])
        sorted_facets = np.argsort(facets)
        mt = mesh.meshtags(self.sim.msh, self.sim.msh.topology.dim-1, facets[sorted_facets], values[sorted_facets])
        ds = ufl.Measure("ds", domain=self.sim.msh, subdomain_data=mt)

        x = ufl.SpatialCoordinate(self.sim.msh)
        δ = 0.1
        σxx_ssa = δ/2 + (x[1]-1)
        t = ufl.as_vector((σxx_ssa, 0))
        
        
        self.F = (ufl.inner(σ, mf.ε(v))\
              - ufl.inner(f, v) 
            #   -p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
            # - p_crack*ufl.inner(ufl.grad(g), v) \
            #  + p_crack* ufl.inner(Iprime*ufl.grad(d), v)\
              ) * ufl.dx 
        
        self.F += (
              + p_w * ufl.inner(n, v) * ufl.ds 
            #   - ufl.inner(t, v) * ds(1) \
            #   + ufl.inner(t,v) * ds(3)\
            #   + p_w * ufl.inner(n,v)*ds(2)
            # 
        )
            
        

        self.J = ufl.derivative(self.F,self.u,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_u)

    def solve(self):
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u_prev_it.x.array[:] = self.u.x.array[:]

    def timestep(self):
        self.u_prev_time.x.array[:] = self.u.x.array[:]


class Elastic3D(Momentum):
    def __init__(self, sim):

        super().__init__(sim)
        

        self.U = fem.functionspace(self.sim.msh, ("Lagrange", 1, (3, )))

        self.u = fem.Function(self.U, name="displacement")
        self.ε_e = mf.ε3d(self.u)
        self.ψplus = self.sim.free_energy_plus(self.ε_e, self.sim.params.ν)

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



        self.F = (ufl.inner(σ, mf.ε3d(v))\
              - ufl.inner(mf.v2to3(f), v) 
              -p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
            # - p_crack*ufl.inner(ufl.grad(g), v) \
            #  + p_crack* ufl.inner(Iprime*ufl.grad(d), v)\
              ) * ufl.dx \
            + p_w * ufl.inner(mf.v2to3(n), v) * ufl.ds 
        

        self.J = ufl.derivative(self.F,self.u,ufl.TrialFunction(self.U))
            
        
        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_u)

    def solve(self):
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u_prev_it.x.array[:] = self.u.x.array[:]

    def timestep(self):
        self.u_prev_time.x.array[:] = self.u.x.array[:]
