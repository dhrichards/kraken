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



class viscoelastic_damage:
    def __init__(self, msh, bc_funcs):
        self.msh = msh
        self.params = parameters.Params_with_uc(self.msh)

      
        self.u_el = bufl.element("CG", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.u_el, self.p_el])

        self.W = fem.functionspace(self.msh, self.mixed_el)
        self.dw = fem.Function(self.W, name="mixed function")

        self.du, self.du_v, self.dp = ufl.split(self.dw)
        self.du_e = self.du - self.du_v

        self.W0 = self.W.sub(0)
        self.W1 = self.W.sub(1)
        self.W2 = self.W.sub(2)

        self.U, _ = self.W0.collapse()
        self.U_v, _ = self.W1.collapse()
        self.P, _ = self.W2.collapse()

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time

        self.dw_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.du_prev_it, self.du_v_prev_it, self.dp_prev_it = ufl.split(self.dw_prev_it)
        self.du_e_prev_it = self.du_prev_it - self.du_v_prev_it

        self.u = self.u_prev_time + self.du
        self.u_v = self.u_v_prev_time + self.du_v
        self.u_e = self.u_e_prev_time + self.du_e
        self.p = self.p_prev_time + self.dp
        
        # self.e_el = bufl.element("DG", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,self.msh.geometry.dim))
        # self.E = fem.functionspace(self.msh, self.e_el)
        # self.ε_e_prev_time = fem.Function(self.E, name="elastic strain tensor")
        # self.ε_e = mf.ε(self.du_e) + self.ε_e_prev_time
        self.ε_e = mf.ε(self.u_e)
    

        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))

        # self.H_el = bufl.quadrature_element(
        #     self.msh.basix_cell(), value_shape=(), scheme="default", degree=1
        # )
        self.H_space = fem.functionspace(self.msh, ("DG", 1))

        self.bc_u = bc_funcs[0](self.W)
        self.bc_d = bc_funcs[1](self.D)

      
        self.d = fem.Function(self.D, name="damage")
        self.g = es.degradation_default(self.d)
        self.d_prev_time = fem.Function(self.D, name="damage previous time")
        self.Hprev = fem.Function(self.H_space, name="history")


    def setup_all(self):
        self.setup()
        damage.setup_damage_bounded(self, lambda d: d, es.free_energy_plus_lo)


    def setup(self):


        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.msh)

        δt = self.params.dtstar
    
    
        η = mf.viscosity(mf.ε(self.du_v_prev_it/δt), self.params.n, 1.e-8)

        p_w = mf.water_pressure(self.msh,self.du,self.params.ucstar) +self.params.patmstar

        p_i = mf.overburden_pressure(self.msh, self.params.ρistar) + self.params.patmstar
    
        δ = ufl.Identity(self.msh.geometry.dim)
        σ0 = es.cauchy_stress(self.ε_e, self.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.params.ν)
        # # σplus = es.stress_plus_dp(self.ε_e, self.params.ν)
        σminus = σ0 - σplus
        # σminus = -mf.overburden_pressure(self.msh, self.params.ρistar)* ufl.Identity(self.msh.geometry.dim)
        # # σminus = -p_ext*δ
        # σplus = σ0 - σminus
        σ = self.g*σplus + σminus
        # σ = self.g*σ0

        # σ = pt.degraded_stress(self.ε_e, 
                            #    mf.ε(self.du_e_prev_it) + mf.ε(self.u_e_prev_time), 
                            #    self.g, self.params.ν)

        

        f = mf.body_force(self.msh, self.params.ρistar)

        # p_deg = g*es.positive_part(-self.p) + es.negative_part(-self.p)
        p_prev_it = self.p_prev_time + self.dp_prev_it
        # p_deg = pt.degraded_scalar(-self.p, -p_prev_it, self.g)
        p_deg = self.g*-self.p

        p_crack = p_i #+ 0.1*p_w

        F = (ufl.inner(σ, mf.ε(v_v))\
            #  - (1-g)*ufl.inner(p_ext, ufl.div(v_v))
              - ufl.inner(f, v_v) 
            #  - p_crack* ufl.inner(ufl.grad(self.g), v_v)\
            # - mf.overburden_pressure(self.msh, self.params.ρistar, self.u, self.params.ucstar)*ufl.inner(ufl.grad(g), v_v)
              ) * ufl.dx \
            + p_w * ufl.inner(n, v_v) * ufl.ds \
            + (
                self.g*η*ufl.inner(mf.ε(self.du_v)/δt, mf.ε(v))\
                + ufl.inner(-self.p, ufl.div(v))  \
            -    ufl.inner(σ, mf.ε(v))
             ) * ufl.dx \
            + (
                - self.g*ufl.inner(ufl.div(self.du_v), q) \
                # - ufl.inner(pt.degraded_scalar(ufl.div(self.du_v),-p_prev_it,self.g), q)\
                # - ufl.inner(g*es.positive_part(ufl.div(dot_u_v)) + es.negative_part(ufl.div(dot_u_v)), q)\
            ) * ufl.dx 

        J = ufl.derivative(F,self.dw,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(F, self.dw, bcs=self.bc_u)

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setType("newtonls")
        # opts = PETSc.Options()
        # opts["snes_type"] = "newtonls"
        # opts["snes_linesearch_type"] = "bt"

        # self.elastic_solver.setFromOptions()

        self.solver.setTolerances(rtol=1.0e-7, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")
 

        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        
        



   
    def update_history(self):

        H = es.history_function(self.ε_e,self.Hprev,
                                self.params.ν,self.params.ψcritstar)

        self.Hprev.interpolate(fem.Expression(H,self.H_space.element.interpolation_points()))

    def solve(self):
        self.solver.solve(None, self.dw.x.petsc_vec)
        self.dw.x.scatter_forward()
        self.dw_prev_it.x.array[:] = self.dw.x.array[:]

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    # def update_strain_tensor(self):
    #     temp = fem.Function(self.E, name="elastic strain tensor")
    #     temp.interpolate(fem.Expression(mf.ε(self.du_e), self.E.element.interpolation_points()))

    #     self.ε_e_prev_time.x.array[:] += temp.x.array[:]
       



    
    def timestep(self):

        # self.update_strain_tensor()
        
        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.ucstar*du.x.array.reshape((-1, self.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.dw.x.array[:]
        self.d_prev_time.x.array[:] = self.d.x.array[:]
        # self.w.x.array[:] = 0.0


        
