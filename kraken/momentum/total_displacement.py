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
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc

class SemiLagrangian(Momentum):
     
    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(3,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)



        self.mixed_el = bufl.mixed_element([self.u_el, self.ε_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        self.w.x.array[:] = 1.0
        self.du, self.vec_dε_e, self.dp = ufl.split(self.w)
        self.dε_e = self.map(self.vec_dε_e)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.du_prev_it, self.vec_dε_e_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        self.dε_e_prev_it = self.map(self.vec_dε_e_prev_it)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.vec_ε_e_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.ε_e_prev_time = self.map(self.vec_ε_e_prev_time)
    
        self.bc_u = self.sim.bc_funcs[0](self.W)

        
        self.u = self.u_prev_time + self.du
        self.ε_e = self.ε_e_prev_time + self.dε_e

        self.p = self.p_prev_time + self.dp
        self.p_prev_it = self.p_prev_time + self.dp_prev_it
        
        self.p_crack = self.crack_pressure(self.du)
        self.pw = self.water_pressure(self.du)

        self.dε_v = mf.ε(self.du) - self.dε_e
        self.dε_v_prev_it = mf.ε(self.du_prev_it) - self.dε_e_prev_it
        
        self.ψplus = self.sim.free_energy_plus(self.ε_e, self.sim.params.ν)

    def map(self, v):
        return ufl.as_tensor([[v[0], v[2]],
                              [v[2], v[1]]])
    
    def inverse_map(self, ε):
        return ufl.as_vector([ε[0,0], ε[1,1], ε[0,1]])


    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, S, q = ufl.split(w_test)
        S = self.map(S)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation_default(self.sim.damage.d,1e-12)
        
        

        # σ0 = es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν)
        σ = self.stress(self.ε_e)
        
        
        
        # g_v = es.degradation_default(self.sim.damage.d,self.sim.params.gv_tol)
        A = mf.rate_factor(self.sim.params.T)/self.sim.params.A0
    
        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)


        η0 = mf.viscosity(self.dε_v_prev_it/self.sim.params.dtstar, self.sim.params.n, 1e-13, A=A)
        g_v = es.degradation_default(self.sim.damage.d, self.sim.params.gv_tol)

        self.F = (
            ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v_v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
              ) * ufl.dx \
            + self.pw * ufl.inner(n, v) * ufl.ds \
        

        self.F+= (
                # η0*ufl.inner(εD, mf.ε(v_v))\
                2*g_v*η0*ufl.inner(self.dε_v/self.sim.params.dtstar, S)\
                - ufl.inner(self.p, ufl.tr(S))  \
            -    ufl.inner(σ, S)
             ) * ufl.dx
        

        self.F += (
                - ufl.div(self.du)*q \
                ) * ufl.dx 
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"
        self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]


    
    def timestep(self):
        # self.vec_ε_e_prev_time.interpolate(fem.Expression(self.inverse_map(self.ε_e), self.E.element.interpolation_points()))
        

        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        




