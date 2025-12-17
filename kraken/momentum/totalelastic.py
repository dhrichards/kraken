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

class SemiLagrangian(Momentum):
     
    def __init__(self, sim):
        super().__init__(sim)

        

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.ε_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        # self.w.x.array[:] = 1.0
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.ε_eD_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.bc_u = self.sim.bc_funcs[0](self.W)

        self.DG0 = fem.functionspace(self.sim.msh, ("DG", 0))
        self.areaf = ufl.TestFunction(self.DG0)
        self.cell_area_form = fem.form(self.areaf * ufl.dx)
        self.area_0 = np.copy(fem.assemble_vector(self.cell_area_form).array)

        self.area_ratio = fem.Function(self.DG0)
        self.area_ratio.x.array[:] = 1.0

        self.du, self.dε_eD, self.dp = ufl.split(self.w)

        self.du_prev_it, self.dε_eD_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        
        self.u = self.u_prev_time + self.du
        self.ε_eD = self.ε_eD_prev_time + self.dε_eD
        self.ε_eD_prev_it = self.ε_eD_prev_time + self.dε_eD_prev_it

        self.vel = self.du / self.sim.params.dtstar

        self.p = self.p_prev_time + self.dp
        self.p_prev_it = self.p_prev_time + self.dp_prev_it

        self.tr_ε_e = -self.p/es.Koverμ(self.sim.params.ν)
        self.tr_ε_e_prev_time = -self.p_prev_time/es.Koverμ(self.sim.params.ν)
        self.tr_ε_e_prev_it = -self.p_prev_it/es.Koverμ(self.sim.params.ν)

        self.ε_e = self.ε_eD + (1/3)*self.tr_ε_e*ufl.Identity(2)
        self.ε_e_prev_time = self.ε_eD_prev_time + (1/3)*self.tr_ε_e_prev_time*ufl.Identity(2)
        
        self.p_crack = self.crack_pressure(self.du)
        self.pw = self.water_pressure(self.du)


        self.ε_e3 = mf.deviatoric2d_to_3d(self.ε_eD) + (1/3)*self.tr_ε_e*ufl.Identity(3)
        self.ψplus = self.sim.free_energy_plus(self.ε_e3,self.sim.params.ν)



    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, S, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation_default(self.sim.damage.d,1e-12)

        # σ3 = self.stress(self.ε_e)
        # σ = mf.tensor_3d_to_2d(σ3)
        # τ = mf.tensor_3d_to_2d(ufl.dev(σ3))

       
        # σ3_prev = self.stress(self.ε_e_prev_time)
        # τ_prev = mf.tensor_3d_to_2d(ufl.dev(σ3_prev))
        # dτ = τ - τ_prev
        σ = self.stress_alt(self.ε_eD, self.tr_ε_e)
        τ = self.deviatoric_stress(self.ε_eD, self.tr_ε_e)
        
        τ_prev = self.deviatoric_stress(self.ε_eD_prev_time, self.tr_ε_e_prev_time)
        dτ = τ - τ_prev

        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        A = mf.rate_factor(self.sim.params.T)/self.sim.params.A0
        τ3d = mf.deviatoric2d_to_3d(2*self.ε_eD)
        τe2 = 0.5*ufl.inner(τ3d,τ3d)

        self.F = (
            ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v_v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
              ) * ufl.dx \
            + self.pw * ufl.inner(n, v) * ufl.ds \
        
        self.F+= (   
                + ufl.inner(τ, S)*A \
                + ufl.inner(dτ/self.sim.params.dtstar, S) \
                - 2*ufl.inner(mf.ε(self.du/self.sim.params.dtstar), S)\
            
             ) * ufl.dx
       
        self.F += (
                - ufl.div(self.du)*q \
                # - ufl.tr(τ0)*q
                ) * ufl.dx 
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"
        self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]

    
        
    def timestep(self):

        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        


        

 



class SemiLagrangian3D(Momentum):
     
    def __init__(self, sim):
        super().__init__(sim)

        # v_el = bufl.element("CG", self.sim.msh.basix_cell(), 1, shape=(3,))
        # self.V = fem.functionspace(self.sim.msh, v_el)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(2,))
        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 2, shape=(4,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.ε_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.vec_ε_e_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.ε_e_prev_time = mf.short_voigt2tensor(self.vec_ε_e_prev_time)

        self.bc_u = self.sim.bc_funcs[0](self.W)

        self.DG0 = fem.functionspace(self.sim.msh, ("DG", 0))
        self.areaf = ufl.TestFunction(self.DG0)
        self.cell_area_form = fem.form(self.areaf * ufl.dx)
        self.area_0 = np.copy(fem.assemble_vector(self.cell_area_form).array)

        self.area_ratio = fem.Function(self.DG0)
        self.area_ratio.x.array[:] = 1.0

        self.du, self.vec_dε_e, self.dp = ufl.split(self.w)
        self.dε_e = mf.short_voigt2tensor(self.vec_dε_e)

        self.du_prev_it, self.vec_dε_e_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        
        self.u = self.u_prev_time + self.du
        self.ε_e = self.ε_e_prev_time + self.dε_e
        self.ε_e_prev_it = self.ε_e_prev_time + mf.short_voigt2tensor(self.vec_dε_e_prev_it)

        self.vel = self.du / self.sim.params.dtstar

        self.p = self.p_prev_time + self.dp
        self.p_prev_it = self.p_prev_time + self.dp_prev_it

        
        self.p_crack = self.crack_pressure(self.du)
        self.pw = self.water_pressure(self.du)

        self.ψplus = self.sim.free_energy_plus(self.ε_e,self.sim.params.ν)



    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, s, q = ufl.split(w_test)
        S = mf.short_voigt2tensor(s)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation_default(self.sim.damage.d,1e-12)


        σ = self.stress(self.ε_e)
        # σ = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        τ = σ + self.p*ufl.Identity(3)

        # dσ = σ - self.stress(self.ε_e_prev_time)
        
        dσ = es.cauchy_stress(self.dε_e, self.sim.params.ν)
        dτ = dσ + self.dp*ufl.Identity(3)
        # self.ρ = mf.ice_density(self.sim.msh,self.sim.params.ρi/self.sim.params.ρw,350/self.sim.params.ρw,32.5/300)/self.area_ratio
        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        A = mf.rate_factor(self.sim.params.T)/self.sim.params.A0

        τe2 = 0.5*ufl.inner(ufl.dev(σ), ufl.dev(σ))

        self.F = (
            ufl.inner(mf.tensor_3d_to_2d(σ), mf.ε(v)) - ufl.inner((f), v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v_v)\
            - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
              ) * ufl.dx \
            + self.pw * ufl.inner((n), v) * ufl.ds \
        
        self.F+= (   
                + ufl.inner(τ, S)*A*τe2 \
                + ufl.inner(dτ/self.sim.params.dtstar, S) \
                - 2*ufl.inner(mf.ε(self.du/self.sim.params.dtstar), mf.tensor_3d_to_2d(S))\
            
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

        
        du = fem.Function(self.V)
        du.interpolate(fem.Expression(mf.v3to2(self.du),self.V.element.interpolation_points()))
        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        


        

 




    


    
