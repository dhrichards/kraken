from .base import Momentum
import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.models import damage
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc


class MixedDisplacement(Momentum):

    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.u_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time

        self.bc_u = self.sim.bc_funcs[0](self.W)


    # def setup_weak_form(self):

    #     w_test = ufl.TestFunction(self.W)
    #     v, v_v, q = ufl.split(w_test)
    #     n = ufl.FacetNormal(self.sim.msh)

    #     g = self.sim.damage.g
    #     η = mf.viscosity(mf.ε(self.vel), self.sim.params.n, 1.e-8)

    #     σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
    #     σplus = es.stress_plus_lo(self.ε_e, self.sim.params.ν)
    #     σminus = σ0 - σplus
    #     σ = g * σplus + σminus

    #     f = mf.body_force(self.sim.msh, self.sim.params.ρistar)









    

class SmallDisplacement(MixedDisplacement):
    
    def __init__(self, sim):
        super().__init__(sim)

        self.w = fem.Function(self.W, name="mixed function")
        self.u, self.u_v, self.p = ufl.split(self.w)
        self.u_e = self.u - self.u_v

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.u_v_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)
        self.u_e_prev_it = self.u_prev_it - self.u_v_prev_it


        self.ε_e = mf.ε(self.u_e)

    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = self.sim.damage.g

        δt = self.sim.params.dtstar

        dot_u_v = (self.u_v - self.u_v_prev_time)/δt
        dot_u_v_prev_it = (self.u_v_prev_it - self.u_v_prev_time)/δt

        η = mf.viscosity(mf.ε(dot_u_v_prev_it), self.sim.params.n, 1.e-8)

        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        σ = g * σplus + σminus

        f = mf.body_force(self.sim.msh, self.sim.params.ρistar)

        x = ufl.SpatialCoordinate(self.sim.msh)
        p_crack = ufl.conditional(ufl.gt(x[0],25.666),1,0)*mf.water_pressure(self.sim.msh, self.u, self.sim.params.ucstar, level=0.05) + self.sim.params.patmstar
        # p_crack = mf.water_pressure(self.sim.msh, level=0.05) + self.sim.params.patmstar
        p_w = mf.water_pressure(self.sim.msh, self.u, self.sim.params.ucstar) + self.sim.params.patmstar


        self.F = (
            ufl.inner(σ, mf.ε(v_v))\
            #  - (1-g)*ufl.inner(p_ext, ufl.div(v_v))
              - ufl.inner(f, v_v) 
             - p_crack* ufl.inner(ufl.grad(g), v_v)\
            # - p_crack*ufl.inner(2*self.d*ufl.grad(self.d), v_v) \
            # - mf.overburden_pressure(self.msh, self.params.ρistar, self.u, self.params.ucstar)*ufl.inner(ufl.grad(g), v_v)
              ) * ufl.dx \
            + p_w * ufl.inner(n, v_v) * ufl.ds \
        
        self.F+= (
                g*η*ufl.inner(mf.ε(dot_u_v), mf.ε(v))\
                + ufl.inner(-self.p, ufl.div(v))  \
            -    ufl.inner(σ, mf.ε(v))
             ) * ufl.dx \
            + (
                - ufl.inner(ufl.div(dot_u_v), q) \
                # - (self.g-mf.degradation_default(self.d_prev_time))*q/self.params.dtstar
                # - ufl.inner(pt.degraded_scalar(ufl.div(dot_u_v),-self.p_prev_it,g), q)\
                # - ufl.inner(g*es.positive_part(ufl.div(dot_u_v)) + es.negative_part(ufl.div(dot_u_v)), q)\
            ) * ufl.dx 
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]

  
    
    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]


        
       

class SemiLagrangian(MixedDisplacement):

    def __init__(self, sim):
        super().__init__(sim)
        

        self.dw = fem.Function(self.W, name="mixed function")

        self.du, self.du_v, self.dp = ufl.split(self.dw)
        self.du_e = self.du - self.du_v

        self.dw_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.du_prev_it, self.du_v_prev_it, self.dp_prev_it = ufl.split(self.dw_prev_it)
        self.du_e_prev_it = self.du_prev_it - self.du_v_prev_it

        self.u = self.u_prev_time + self.du
        self.u_v = self.u_v_prev_time + self.du_v
        self.u_e = self.u_e_prev_time + self.du_e
        self.p = self.p_prev_time + self.dp

        self.ε_e = mf.ε(self.u_e)


    def setup_momentum(self):
        
        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)
        g = self.sim.damage.g

        δt = self.sim.params.dtstar
    
    
        η = mf.viscosity(mf.ε(self.du_v_prev_it/δt), self.sim.params.n, 1.e-8)

        p_w = mf.water_pressure(self.sim.msh,self.du,self.sim.params.ucstar) +self.sim.params.patmstar
        # p_crack = mf.water_pressure_static(self.sim.msh,level=0.0) + self.sim.params.patmstar
        p_crack = p_w

        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        σ = g*σplus + σminus

        

        f = mf.body_force(self.sim.msh, self.sim.params.ρistar)
        

        
        self.F = (ufl.inner(σ, mf.ε(v_v))\
              - ufl.inner(f, v_v) 
            #  - p_crack* ufl.inner(ufl.grad(g), v_v)\
            - p_crack*ufl.inner(ufl.Dx(g, 0), v[0]) \
                ) * ufl.dx \
            + p_w * ufl.inner(n, v_v) * ufl.ds \
            + (
                g*η*ufl.inner(mf.ε(self.du_v)/δt, mf.ε(v))\
                + ufl.inner(-self.p, ufl.div(v))  \
            -    ufl.inner(σ, mf.ε(v))
             ) * ufl.dx \
            + (
                - ufl.inner(ufl.div(self.du_v), q) \
                  ) * ufl.dx 

        self.J = ufl.derivative(self.F,self.dw,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.dw, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.dw.x.petsc_vec)
        self.dw.x.scatter_forward()
        self.dw_prev_it.x.array[:] = self.dw.x.array[:]

  
    
    def timestep(self):

        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.dw.x.array[:]

        
        
    