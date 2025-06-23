import numpy as np
from dolfinx import fem
from mpi4py import MPI
import basix.ufl as bufl
import ufl
import numpy as np
from kraken.models import damage
from kraken.numerics import maths_functions as mf
from kraken.numerics import total_velocity_maths as mt
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc

class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, params):
        self.msh = msh
        self.params = params

        
        self.u_el = bufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
        self.ε_el = bufl.element("CG", msh.basix_cell(), 2, shape=(2,2))
        self.p_el = bufl.element("CG", msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.ε_el, self.p_el])

        self.W = fem.functionspace(msh, self.mixed_el)
        self.w = fem.Function(self.W, name="mixed function")
        self.w.x.array[:] = 1.0

        self.u, self.σD, self.p = ufl.split(self.w)

        self.W0 = self.W.sub(0)
        self.W1 = self.W.sub(1)
        self.W2 = self.W.sub(2)

        self.U, _ = self.W0.collapse()
        self.U_v, _ = self.W1.collapse()
        self.P, _ = self.W2.collapse()

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.σD_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.σD_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)

        self.ε_e = mt.elastic_strain(self.σD, self.p, self.params.ν)
        
        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))

        # self.H_el = bufl.quadrature_element(
        #     self.msh.basix_cell(), value_shape=(), scheme="default", degree=2
        # )
        self.H_space = fem.functionspace(self.msh, ("DG", 1))

        self.bc_u = bc_funcs[0](self.W)
        self.bc_d = bc_funcs[1](self.D)

      
        self.d = fem.Function(self.D, name="damage")
        self.d_prev_time = fem.Function(self.D, name="damage previous time")
        self.Hprev = fem.Function(self.H_space, name="history")
        self.H = mf.history_function(self.ε_e, self.Hprev,
                                    self.params.ν, self.params.ψcritstar)

    def setup(self):
        self.setup_displacement()
        damage.setup_damage_non_linear(self)

   

    def update_history(self):
        self.Hprev.interpolate(fem.Expression(self.H,self.H_space.element.interpolation_points()))

    def setup_displacement(self):

        δt = self.params.dtstar
        λoverμ = self.params.λ/self.params.μ
        D = self.msh.geometry.dim

        w_test = ufl.TestFunction(self.W)
        v, τ, q = ufl.split(w_test)

        g = mf.degradation_default(self.d, 1e-3)
    
        n = ufl.FacetNormal(self.msh)
        
        p_ext = mf.water_pressure(self.msh,self.u,self.params.ucstar*self.params.dtstar) +self.params.patmstar
        f = mf.body_force(self.msh, self.params.ρistar, self.params.slope_angle)


        dot_σD = (self.σD - self.σD_prev_time)/δt \
            + self.params.ucstar*mt.ucm_steady(self.σD, self.u)/δt
        dot_p = (self.p - self.p_prev_time)/δt

        
        τe2 = ufl.inner(self.σD, self.σD) / 2
        η = 0.5 * (τe2 + 1e-7)**(1-self.params.n)
        
    
        F = (
            ufl.inner(g*self.σD, mf.ε(v)) \
            - ufl.inner(g*self.p,ufl.div(v)) \
            - ufl.inner(g*f, v)\
            - p_ext* ufl.inner(ufl.grad(g), v)\
        ) * ufl.dx \
        + g * p_ext * ufl.inner(n, v) * ufl.ds \
        + ( 
            ufl.inner(self.σD, τ) \
            + ufl.inner(dot_σD, τ) \
            - 2*η*ufl.inner(mf.ε(self.u), τ) \
        ) * ufl.dx \
        + (
            - ufl.inner(ufl.div(self.u), q) \
            - (1.0/(D*(λoverμ + 2/D)))*dot_p*q
        ) * ufl.dx


        J = ufl.derivative(F, self.w, ufl.TrialFunction(self.W))

        self.problem = solvers.SNESProblem(F, self.w, bcs=self.bc_u)

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setType("newtonls")
        # opts = PETSc.Options()
        # opts["snes_type"] = "newtonls"
        # opts["snes_linesearch_type"] = "bt"

        # self.elastic_solver.setFromOptions()

        self.solver.setTolerances(rtol=1.0e-8, max_it=100)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-8)
        self.solver.getKSP().getPC().setType("lu")
        # self.solver.getKSP().getPC().setFactorSolverType("mumps")
 

        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def solve_displacement(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w_prev_it.x.array[:] = self.w.x.array[:]
        
        

    def fixed_point(self, max_its=100, tol=1e-4, min_its=2, solve_damage=True):
        L2_old = 0.0

        one = fem.Function(self.D)
        one.x.array[:] = 1.0
        area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

        area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))


        
        for i in range(max_its):
            
            if solve_damage:
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
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.ucstar*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        self.u.x.array[:] = 0.0
        self.u_prev_it.x.array[:] = 0.0

    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_it.x.array[:] = 0.0



