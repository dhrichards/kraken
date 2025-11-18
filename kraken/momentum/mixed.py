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
        self.w.x.array[:] = 1.0
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time

        self.w_prev_2 = fem.Function(self.W, name="mixed function 2 timesteps previous")
        self.u_prev_2, self.u_v_prev_2, self.p_prev_2 = ufl.split(self.w_prev_2)

        
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

        
        A = mf.rate_factor(self.sim.T)/self.sim.params.A
        η = mf.viscosity(mf.εD(self.vel_prev_it), self.sim.params.n, 1.e-15, A=A)
      
        
        σ = self.stress(self.ε_e)
        
        # σD0 = 2*mf.dev3(self.ε_e_prev_it)
        # η = mf.viscosity_stress(σD0, self.sim.params.n, 1.e-14, A=A)

       
     

        g_v = es.degradation_default(self.sim.damage.d,1e-5)
        
        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        Iprime = 2*self.sim.damage.d
        # Iprime = 1.0
        self.F = (
            0.5*self.sim.params.C_inertia*ufl.inner(self.accel, v)  \
            + ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
            # + self.p_crack*Iprime*ufl.inner(ufl.Dx(self.sim.damage.d,0), v[0]) \
            # + self.p_crack*Iprime*ufl.inner(ufl.grad(self.sim.damage.d), v)
              ) * ufl.dx \
            + self.pw * ufl.inner(n, v) * ufl.ds \
        
        self.F+= (
                g_v*η*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
                + ufl.inner(-self.p, ufl.div(v_v))  \
            -    ufl.inner(σ, mf.ε(v_v))
             ) * ufl.dx
       
       
        self.F += (
                - ufl.inner(ufl.div(self.vel), q) \
                # - (self.p-self.p_prev_time)*q/self.sim.params.dtstar
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
        self.ε_e_prev_it = mf.ε(self.u_e_prev_it)
        self.vel = self.du_v/self.sim.params.dtstar
        self.vel_prev_it = self.du_v_prev_it/self.sim.params.dtstar

        self.pw = self.water_pressure(self.du)
        self.p_crack = self.crack_pressure(self.du)

        self.accel = (self.u - 2*self.u_prev_time + self.u_prev_2)/(self.sim.params.dtstar**2)


        

    
    def timestep(self):

        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_2.x.array[:] = self.w_prev_time.x.array[:]
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0




class SemiLagrangianEpsilon(SemiLagrangian):

    def __init__(self, sim):
        super().__init__(sim)

        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsilon previous time")
        self.ε_e = mf.ε(self.du_e) + self.ε_e_prev_time
        self.ε_e_prev_it = mf.ε(self.du_e_prev_it) + self.ε_e_prev_time

    def timestep(self):
        super().timestep()
        self.ε_e_prev_time.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))
        
        
    