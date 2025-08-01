import numpy as np
from dolfinx import fem, default_scalar_type
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken import parameters
from kraken.models import damage
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
        self.w = fem.Function(self.W, name="mixed function")

        self.u, self.u_v, self.p = ufl.split(self.w)
        self.u_e = self.u - self.u_v
        self.ε_e = mf.ε(self.u_e)

        self.W0 = self.W.sub(0)
        self.W1 = self.W.sub(1)
        self.W2 = self.W.sub(2)

        self.U, _ = self.W0.collapse()
        self.U_v, _ = self.W1.collapse()
        self.P, _ = self.W2.collapse()

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.u_v_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)
        self.u_e_prev_it = self.u_prev_it - self.u_v_prev_it

        self.D = fem.functionspace(self.msh, ("Lagrange", 1))

        # self.H_el = bufl.quadrature_element(
        #     self.msh.basix_cell(), value_shape=(), scheme="default", degree=1
        # )
        self.H_space = fem.functionspace(self.msh, ("DG", 1))

        self.bc_u = bc_funcs[0](self.W)
        self.bc_d = bc_funcs[1](self.D)

      
        self.d = fem.Function(self.D, name="damage")
        # self.g = mf.degradation_Lo2023(self.d, 5)
        self.g = es.degradation_default(self.d)
        self.d_prev_time = fem.Function(self.D, name="damage previous time")
        self.Hprev = fem.Function(self.H_space, name="history")

        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))

       
       
     
    def setup_all(self):
        self.setup_momentum()
        # self.setup_damage()
        # damage.setup_damage_non_linear(self)
        damage.setup_damage_bounded(self, lambda d: d, es.free_energy_plus_lo)


    def setup_momentum(self):


        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        

        dot_u_v = (self.u_v - self.u_v_prev_time)/self.params.dtstar
        dot_u_v_prev_it = (self.u_v_prev_it - self.u_v_prev_time)/self.params.dtstar
        

        η = mf.viscosity(mf.ε(dot_u_v_prev_it), self.params.n, 1.e-8)

        σ0 = es.cauchy_stress(self.ε_e, self.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.params.ν)
        σminus = σ0 - σplus
        σ = self.g*σplus + σminus
        # σ = self.g*σ0

        # σ = pt.degraded_stress(self.ε_e, mf.ε(self.u_e_prev_it), self.g, self.params.ν)

        p_w = mf.water_pressure(self.msh,self.u,self.params.ucstar) +self.params.patmstar
        p_i = mf.overburden_pressure(self.msh, self.params.ρistar) + self.params.patmstar

        f = mf.body_force(self.msh, self.params.ρistar)






        # p_deg = g*es.positive_part(-self.p) + es.negative_part(-self.p)
        # p_deg = pt.degraded_scalar(-self.p, -self.p_prev_it, g)
        p_deg = self.g*-self.p
        # p_deg = -self.p
        n = ufl.FacetNormal(self.msh)


        F = (
            ufl.inner(σ, mf.ε(v_v))\
            #  - (1-g)*ufl.inner(p_ext, ufl.div(v_v))
              - ufl.inner(f, v_v) 
            #  - p_i* ufl.inner(ufl.grad(self.g), v_v)\
            # - mf.overburden_pressure(self.msh, self.params.ρistar, self.u, self.params.ucstar)*ufl.inner(ufl.grad(g), v_v)
              ) * ufl.dx \
            + p_w * ufl.inner(n, v_v) * ufl.ds \
        
        F+= (
                self.g*η*ufl.inner(mf.ε(dot_u_v), mf.ε(v))\
                + ufl.inner(-self.p, ufl.div(v))  \
            -    ufl.inner(σ, mf.ε(v))
             ) * ufl.dx \
            + (
                - ufl.inner(ufl.div(dot_u_v), q) \
                # - (self.g-mf.degradation_default(self.d_prev_time))*q/self.params.dtstar
                # - ufl.inner(pt.degraded_scalar(ufl.div(dot_u_v),-self.p_prev_it,g), q)\
                # - ufl.inner(g*es.positive_part(ufl.div(dot_u_v)) + es.negative_part(ufl.div(dot_u_v)), q)\
            ) * ufl.dx 
        

        J = ufl.derivative(F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(F, self.w, bcs=self.bc_u)

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setType("newtonls")
        # opts = PETSc.Options()
        # opts["snes_type"] = "newtonls"
        # opts["snes_linesearch_type"] = "bt"

        # self.elastic_solver.setFromOptions()

        self.solver.setTolerances(rtol=1.0e-7, max_it=50)
        self.solver.getKSP().setType("preonly")
        # #non zero initial guess
        # self.solver.getKSP().setInitialGuessNonzero(True)
        # self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        # self.solver.getKSP().getPC().subPCType.setType("ilu")
        # # self.solver.getKSP().getPC().setFieldSplitType(1)
        self.solver.getKSP().getPC().setFactorSolverType("mumps")
        
 

        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        
    
    def update_history(self):

        H = es.history_function(self.ε_e,self.Hprev,
                                self.params.ν, self.params.ψcritstar)
        self.Hprev.interpolate(fem.Expression(H,self.H_space.element.interpolation_points()))

        

    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)


    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.d_prev_time.x.array[:] = self.d.x.array[:]
        # self.d_prev_time.x.array[:] = self.d.x.array[:]
        # self.w.x.array[:] = 0.0
