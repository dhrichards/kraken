import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.models import damage
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc



class elastic_damage:
    def __init__(self, msh, bc_funcs, params):
        self.msh = msh
        self.params = params

        self.u_el = bufl.element("CG", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.p_el])

        self.W = fem.functionspace(self.msh, self.mixed_el)
        self.w = fem.Function(self.W, name="mixed function")

        self.u, self.p = ufl.split(self.w)
        self.ε_e = mf.ε(self.u)

        self.W0 = self.W.sub(0)
        self.W1 = self.W.sub(1)

        self.U, _ = self.W0.collapse()
        self.P, _ = self.W1.collapse()

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)
       
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))

        # self.H_el = bufl.quadrature_element(
        #     self.msh.basix_cell(), value_shape=(), scheme="default", degree=1
        # )
        self.H_space = fem.functionspace(self.msh, ("DG", 1))

        self.bc_u = bc_funcs[0](self.W)
        self.bc_d = bc_funcs[1](self.D)

      
        self.d = fem.Function(self.D, name="damage")
        # self.g = mf.degradation_Lo2023(self.d, 5)
        self.g = mf.degradation_default(self.d)
        self.d_prev_time = fem.Function(self.D, name="damage previous time")
        self.Hprev = fem.Function(self.H_space, name="history")

        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        
     
    def setup_all(self):
        self.setup_momentum()
        # self.setup_damage()
        damage.setup_damage_non_linear(self)
        # damage.setup_damage_bounded(self, lambda d: d**2, es.free_energy_plus_spectral)


    def setup_momentum(self):


        w_test = ufl.TestFunction(self.W)
        v, q = ufl.split(w_test)
        
        # σ0 = es.cauchy_stress(self.ε_e, self.params.ν)

        δ = ufl.Identity(self.msh.geometry.dim)
        λstar = self.params.λ/self.params.μ
        k = λstar + 2/3
        
        # σ = pt.degraded_stress(self.ε_e, mf.ε(self.u_e_prev_it), self.g, self.params.ν)

        p_ext = mf.water_pressure(self.msh,self.u,self.params.ucstar) +self.params.patmstar

        f = self.g*mf.body_force(self.msh, self.params.ρistar)






        # p_deg = g*es.positive_part(-self.p) + es.negative_part(-self.p)
        # p_deg = pt.degraded_scalar(-self.p, -self.p_prev_it, g)
        p_deg = self.g*-self.p
        # p_deg = -self.p
        n = ufl.FacetNormal(self.msh)

        # ψ = es.free_energy(self.ε_e, self.params.ν)
        # ψplus = es.free_energy_plus_dp(self.ε_e, self.params.ν)
        # ψminus = ψ - ψplus
        # elastic_energy = (\
        #     self.g*ψ
        #     # self.g*ψplus + ψminus \
        #     - ufl.dot(f, self.u) \
        #     - p_ext*ufl.inner(ufl.grad(self.g), self.u)\
        #      )* ufl.dx \
        #     + self.g * p_ext *  ufl.dot(n, self.u) * ufl.ds
        
        # F = ufl.derivative(elastic_energy,self.u,v_v)

        εD = mf.ε(self.u) - ufl.tr(mf.ε(self.u))*δ/3
        F = (
            ufl.inner(pt.degraded_deviatoric(εD,mf.ε(self.u_prev_it),self.g,self.params.ν), mf.ε(v))\
            # ufl.inner(self.g*2*εD, mf.ε(v))\
            #  - self.g*ufl.inner(self.p, ufl.div(v)) \
            + ufl.inner(pt.degraded_scalar(-self.p,-self.p_prev_it, self.g), ufl.div(v)) \
            #  - (1-g)*ufl.inner(p_ext, ufl.div(v_v))
              - ufl.inner(f, v) 
             - p_ext* ufl.inner(ufl.grad(self.g), v)\
                 ) * ufl.dx \
            + self.g*p_ext * ufl.inner(n, v) * ufl.ds \
            - ( 
            # self.g*k*ufl.inner(ufl.div(self.u), q) \
            k*ufl.inner(pt.degraded_scalar(ufl.div(self.u), -self.p_prev_it, self.g),q) \
            +self.p*q\
              )* ufl.dx
        
      
        

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

        H = mf.history_function(self.ε_e,self.Hprev,
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
