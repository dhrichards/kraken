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

        self.w = fem.Function(self.W, name="mixed function")
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time

        self.bc_u = self.sim.bc_funcs[0](self.W)

        self.DG0 = fem.functionspace(self.sim.msh, ("DG", 0))
        self.areaf = ufl.TestFunction(self.DG0)
        self.cell_area_form = fem.form(self.areaf * ufl.dx)
        self.area_0 = np.copy(fem.assemble_vector(self.cell_area_form).array)

        self.area_ratio = fem.Function(self.DG0)
        self.area_ratio.x.array[:] = 1.0



    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = self.sim.damage.g

        

        η = mf.viscosity(mf.ε(self.vel_prev_it), self.sim.params.n, 1.e-8)

        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        σ = g * σplus + σminus

        # self.ρ = mf.ice_density(self.sim.msh,self.sim.params.ρi/self.sim.params.ρw,350/self.sim.params.ρw,32.5/300)/self.area_ratio
        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        
        self.F = (
            ufl.inner(σ, mf.ε(v_v)) - ufl.inner(f, v_v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v_v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v_v[0]) \
              ) * ufl.dx \
            + self.pw * ufl.inner(n, v_v) * ufl.ds \
        
        self.F+= (
                g*η*ufl.inner(mf.ε(self.vel), mf.ε(v))\
                + ufl.inner(-self.p, ufl.div(v))  \
            -    ufl.inner(σ, mf.ε(v))
             ) * ufl.dx
        
        self.F += (
                - ufl.inner(ufl.div(self.vel), q) \
                ) * ufl.dx 
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"
        # self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]




    

class SmallDisplacement(MixedDisplacement):
    
    def __init__(self, sim):
        super().__init__(sim)

        self.u, self.u_v, self.p = ufl.split(self.w)
        self.u_e = self.u - self.u_v

        self.u_prev_it, self.u_v_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)
        self.u_e_prev_it = self.u_prev_it - self.u_v_prev_it


        self.ε_e = mf.ε(self.u_e)
        self.vel = (self.u_v-self.u_v_prev_time)/self.sim.params.dtstar
        self.vel_prev_it = (self.u_v_prev_it-self.u_v_prev_time)/self.sim.params.dtstar

        self.pw = self.water_pressure(self.u)
        self.p_crack = self.crack_pressure(self.u)
    
  
    
    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]


        
       

class SemiLagrangian(MixedDisplacement):

    def __init__(self, sim):
        super().__init__(sim)
        

        self.du, self.du_v, self.dp = ufl.split(self.w)
        self.du_e = self.du - self.du_v

        self.du_prev_it, self.du_v_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        self.du_e_prev_it = self.du_prev_it - self.du_v_prev_it

        self.u = self.u_prev_time + self.du
        self.u_v = self.u_v_prev_time + self.du_v
        self.u_e = self.u_e_prev_time + self.du_e
        self.p =  self.p_prev_time + self.dp

        self.u_prev_it = self.u_prev_time + self.du_prev_it
        self.u_v_prev_it = self.u_v_prev_time + self.du_v_prev_it
        self.u_e_prev_it = self.u_e_prev_time + self.du_e_prev_it

        self.ε_e = mf.ε(self.u_e)
        self.vel = self.du/self.sim.params.dtstar
        self.vel_prev_it = self.du_v_prev_it/self.sim.params.dtstar

        self.pw = self.water_pressure(self.du)
        self.p_crack = self.crack_pressure(self.du)

        

    
    def timestep(self):

        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        
        
    