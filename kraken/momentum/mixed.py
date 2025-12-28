import adios4dolfinx
from .base import Momentum
import numpy as np
from dolfinx import fem
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
from kraken.numerics.invariants import matrix_function
from kraken.numerics import deviatoric_stress_split as dss


class MixedDisplacement(Momentum):

    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.u_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        # self.w.x.array[:] =1.0
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time

        self.w_start = fem.Function(self.W, name="mixed function at start of iteration")
        self.w_prev_it_start = fem.Function(self.W, name="mixed function previous iteration at start of iteration")

        self.w_prev_2 = fem.Function(self.W, name="mixed function 2 timesteps previous")
        self.u_prev_2, self.u_v_prev_2, self.p_prev_2 = ufl.split(self.w_prev_2)

        self.vel_prev_time = (self.u_v_prev_time - self.u_v_prev_2)/self.sim.params.dtstar
        
        self.bc_u = self.sim.bc_funcs[0](self.W)

        


    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation_default(self.sim.damage.d,self.sim.params.ge_tol)
        

        # σ0 = es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν)
        σ = self.stress(self.ε_e)
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        # σ = 1.5*es.λoverμ(self.sim.params.ν)*ufl.tr(self.ε_e)*ufl.Identity(self.sim.msh.geometry.dim) + 2*self.ε_e
        
        
        
        # g_v = es.degradation_default(self.sim.damage.d,self.sim.params.gv_tol)
        A = mf.rate_factor(self.sim.params.T)/self.sim.params.A0
        
        # η0 = mf.viscosity_stress(es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν), self.sim.params.n, 0, A=A)
        ϵD = ufl.dev(mf.ε(self.vel_prev_it))
        ϵe2 = 0.5*ufl.inner(ϵD,ϵD) + 1e-12
        # η0 = A**(-1/3)*ϵe2**((1-3)/3)/2
        η0 = mf.viscosity(ufl.dev(mf.ε(self.vel_prev_it)), self.sim.params.n, 1e-13, A=A)

        # g_v = ufl.conditional(self.sim.damage.d > 0.98, 0.0, 1.0)

        g_v = es.degradation_default(self.sim.damage.d, self.sim.params.gv_tol)


        η = g*η0 + (1-g)*self.sim.params.gv_tol

        
        τ0 = 2*ufl.dev(self.ε_e)
        τe2 = 0.5*ufl.inner(τ0,τ0) + 1e-7
        # η = g_v/(A*τe2)

        # τe2 = 0.5*(ufl.inner(ufl.dev(σ), ufl.dev(σ))) + 1e-8
        # η = g/(A*τe2) + (1-g)*self.sim.params.gv_tol

        
        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        Iprime = 2*self.sim.damage.d


        # Iprime = 1.0
        self.F = (
            # 0.5*self.sim.params.C_inertia*ufl.inner(self.accel, v)  \
            + ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
            # + self.p_crack*Iprime*ufl.inner(ufl.Dx(self.sim.damage.d,0), v[0]) \
            # + self.p_crack*Iprime*ufl.inner(ufl.grad(self.sim.damage.d), v)
              ) * ufl.dx \
            + self.pw * ufl.inner(n, v) * ufl.ds \
        
        self.F+= (
                # η0*ufl.inner(εD, mf.ε(v_v))\
                2*g*η0*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
                - g*ufl.inner(self.p, ufl.div(v_v))  \
            -    g*ufl.inner(σ0, mf.ε(v_v))
             ) * ufl.dx
        # self.F += (
        #         2*η*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
        #         + ufl.inner(-self.p, ufl.div(v_v))  \
        #         - ufl.inner(σ, mf.ε(v_v))\
        #             ) * ufl.dx
        # self.F += (
        #         ufl.inner(σv0, mf.ε(v_v))\
        #         + ufl.inner(σ, mf.ε(v_v))\
        # )         * ufl.dx


        self.F += (
                - g*ufl.div(self.du)*q \
                ) * ufl.dx 
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        # assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"

        # if self.solver.getConvergedReason() < 0:
        #     print("Did not converge, setting gv_tol to ", self.gv_tol*10)
        #     self.gv_tol = self.gv_tol*10
        #     self.setup_solver()
        #     self.w_prev_it.x.array[:] = self.w_prev_time.x.array[:]
        #     self.w.x.array[:] = self.w_prev_time.x.array[:]
        #     self.sim.damage.w.x.array[:] = self.sim.damage.w_prev_time.x.array[:]
        # # 
        # else:
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


    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]
        self.w_prev_it.x.array[:] = 0.0


        
       

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
        self.p_prev_it = self.p_prev_time + self.dp_prev_it

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
        

        self.w_start.x.array[:] = self.w.x.array[:]
        self.w_prev_it_start.x.array[:] = self.w_prev_it.x.array[:]

    def revert(self):
        self.w.x.array[:] = self.w_start.x.array[:]
        self.w_prev_it.x.array[:] = self.w_prev_it_start.x.array[:]

        
        




class SemiLagrangianEpsilon(SemiLagrangian):

    def __init__(self, sim):
        super().__init__(sim)

        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsiloneprevtime")
        self.ε_e = mf.ε(self.du_e) + self.ε_e_prev_time
        self.ε_e_prev_it = mf.ε(self.du_e_prev_it) + self.ε_e_prev_time

        self.ε_eD = self.ε_e - (1/3)*1.5*ufl.tr(self.ε_e)*ufl.Identity(self.sim.msh.geometry.dim)


        self.ψplus = self.sim.free_energy_plus(self.ε_e,self.sim.params.ν)

    def timestep(self):
        self.ε_e_prev_time.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))

        super().timestep()
        
    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

        
        
    