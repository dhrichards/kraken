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

class TotalDisplacement(Momentum):
     
    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(2,2))

        self.mixed_el = bufl.mixed_element([self.u_el, self.ε_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.ε_v_prev_time = ufl.split(self.w_prev_time)

        self.bc_u = self.sim.bc_funcs[0](self.W)

        self.DG0 = fem.functionspace(self.sim.msh, ("DG", 0))
        self.areaf = ufl.TestFunction(self.DG0)
        self.cell_area_form = fem.form(self.areaf * ufl.dx)
        self.area_0 = np.copy(fem.assemble_vector(self.cell_area_form).array)

        self.area_ratio = fem.Function(self.DG0)
        self.area_ratio.x.array[:] = 1.0


    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, τ = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = self.sim.damage.g

        

        η = mf.viscosity(self.dε_v_prev_it/self.sim.params.dtstar, self.sim.params.n, 1.e-8)

        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        σ = g * σplus + σminus

        # self.ρ = mf.ice_density(self.sim.msh,self.sim.params.ρi/self.sim.params.ρw,350/self.sim.params.ρw,32.5/300)/self.area_ratio
        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        
        self.F = (
            ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v_v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
              ) * ufl.dx \
            + self.pw * ufl.inner(n, v) * ufl.ds \
        
        self.F += (
                g*η*ufl.inner(self.dε_v/self.sim.params.dtstar, τ)\
            -    ufl.inner(σ, τ)
             ) * ufl.dx
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"
        # self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]

        

class SmallDisplacement(TotalDisplacement):
    def __init__(self, sim):
        super().__init__(sim)

        self.u, self.ε_v = ufl.split(self.w)
        self.u_prev_it, self.ε_v_prev_it = ufl.split(self.w_prev_it)
        self.ε_e = mf.ε(self.u) - self.ε_v

        self.p_crack = self.crack_pressure(self.u)
        self.pw = self.water_pressure(self.u)

        self.dε_v = self.ε_v - self.ε_v_prev_time
        self.dε_v_prev_it = self.ε_v_prev_it - self.ε_v_prev_time

    


    
    def timestep(self):

        self.w_prev_time.x.array[:] += self.w.x.array[:]

     
class SemiLagrangian(TotalDisplacement):

    def __init__(self, sim):
        super().__init__(sim)

        self.du, self.dε_v = ufl.split(self.w)

        
        self.du_prev_it, self.dε_v_prev_it= ufl.split(self.w_prev_it)
        
        self.u = self.u_prev_time + self.du
        self.ε_v = self.ε_v_prev_time + self.dε_v
        self.ε_e = mf.ε(self.u) - self.ε_v

        

        
        self.p_crack = self.crack_pressure(self.du)
        self.pw = self.water_pressure(self.du)

    


    
    def timestep(self):

        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        




