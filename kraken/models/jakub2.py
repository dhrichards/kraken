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



class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, params):
        self.msh = msh
        self.params = params

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
        self.Hprev = fem.Function(self.H_space, name="history")
        
     
    def setup(self):
        self.setup_displacement()
        # self.setup_damage()
        # damage.setup_damage_non_linear(self)
        damage.setup_damage_bounded(self, lambda d: d)


    def setup_displacement(self):


        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        

        dot_u_v = (self.u_v - self.u_v_prev_time)/self.params.dtstar
        dot_u_v_prev_it = (self.u_v_prev_it - self.u_v_prev_time)/self.params.dtstar
        

        g = mf.degradation_default(self.d)
        η = mf.viscosity(mf.ε(dot_u_v_prev_it), self.params.n, 1.e-8)

        σ0 = es.cauchy_stress(self.ε_e, self.params.ν)
        # σplus = es.stress_plus_spectral(mf.ε(self.u_e), self.params.ν)
        # σminus = σ0 - σplus
        # σ = g*σplus + σminus
        σ = g*σ0

        # σ = pt.degraded_stress(self.ε_e, mf.ε(self.u_e_prev_it), g, self.params.ν)

        p_ext = mf.water_pressure(self.msh,self.u,self.params.ucstar) +self.params.patmstar
        fi = mf.body_force(self.msh, self.params.ρistar, self.params.slope_angle)
        fw = mf.water_body_force(self.msh)
        f = g*fi #+ (1-g)*fw
        # f = fi


        # p_deg = g*es.positive_part(-self.p) + es.negative_part(-self.p)
        # p_deg = pt.degraded_scalar(-self.p, -self.p_prev_it, g)
        p_deg = g*-self.p
        n = ufl.FacetNormal(self.msh)

        F = (ufl.inner(σ, mf.ε(v_v))\
            #  - (1-g)*ufl.inner(p_ext, ufl.div(v_v))
              - ufl.inner(f, v_v) 
             - p_ext* ufl.inner(ufl.grad(g), v_v)\
              ) * ufl.dx \
            + g*p_ext * ufl.inner(n, v_v) * ufl.ds \
            + (
                g*η*ufl.inner(mf.ε(dot_u_v), mf.ε(v))\
                + ufl.inner(p_deg, ufl.div(v))  \
            -    ufl.inner(σ, mf.ε(v))
             ) * ufl.dx \
            + (
                - ufl.inner(g*ufl.div(dot_u_v), q) \
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
        self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        # self.solver.getKSP().getPC().setFactorSolverType("mumps")
 

        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        
    
    def update_history(self):

        H = mf.history_function(self.ε_e,self.Hprev,
                                self.params.ν, self.params.ψcritstar)
        self.Hprev.interpolate(fem.Expression(H,self.H_space.element.interpolation_points()))

        

    def solve_displacement(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        self.w_prev_it.x.array[:] = self.w.x.array[:]

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def fixed_point(self, max_its=100, tol=1e-4, min_its=2):
        L2_old = 0.0

        one = fem.Function(self.D)
        one.x.array[:] = 1.0
        area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

        area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))


        
        for i in range(max_its):
            
            self.solve_damage()
            self.solve_displacement()
   
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)/area
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")

            if i>min_its-1:
                if error_L2 < tol:
                    break

            L2_old = L2

        # Update history function as finished fixed point iteration
        self.update_history()
        


    

    
    def lagrangian_update(self):
        
        uhh = fem.Function(self.V)
        uhh.interpolate(fem.Expression(self.u - self.u_prev_time,self.V.element.interpolation_points()))
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.ucstar*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        # self.w.x.array[:] = 0.0
        
