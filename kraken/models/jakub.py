import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.models import surface
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc



class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, params, g=mf.degradation_default):
        self.msh = msh
        self.params = params

        self.free_energy_plus = es.free_energy_plus_spectral


        self.u_el = bufl.element("CG", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.u_el, self.p_el])

        self.W = fem.functionspace(self.msh, self.mixed_el)
        self.w = fem.Function(self.W, name="mixed function")

        self.u, self.u_v, self.p = ufl.split(self.w)
        self.u_e = self.u - self.u_v

        self.W0 = self.W.sub(0)
        self.W1 = self.W.sub(1)
        self.W2 = self.W.sub(2)

        self.U, _ = self.W0.collapse()
        self.U_v, _ = self.W1.collapse()
        self.P, _ = self.W2.collapse()

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)

        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))

        # self.H_el = bufl.quadrature_element(
        #     self.msh.basix_cell(), value_shape=(), scheme="default", degree=1
        # )
        self.H_space = fem.functionspace(self.msh, ("DG", 1))

        self.bc_u = bc_funcs[0](self.W)
        self.bc_d = bc_funcs[1](self.D)

      
        self.d = fem.Function(self.D, name="damage")
        self.Hprev = fem.Function(self.H_space, name="history")

        self.g = g(self.d)
        
     
        self.move_mesh = self.lagrangian_update


    def setup_displacement(self):


        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        

        du_v = (self.u_v - self.u_v_prev_time)/self.params.dtstar


        η = mf.viscosity(mf.ε(du_v), self.params.n, 1.e-8)

        σ0 = es.cauchy_stress(mf.ε(self.u_e), self.params.ν)
        σplus = es.stress_plus_spectral(mf.ε(self.u_e), self.params.ν)
        σminus = σ0 - σplus
        σ = self.g*σplus + σminus
        # σ = self.g*σ0

        p_ext = mf.water_pressure(self.msh,self.u,self.params.uc_star) +self.params.patmstar
        f = self.g*mf.body_force(self.msh, self.params.ρistar, self.params.slope_angle)


        p_deg = self.g*es.positive_part(-self.p) + es.negative_part(-self.p)
        n = ufl.FacetNormal(self.msh)

        F = (ufl.inner(σ, mf.ε(v_v))\
              - ufl.inner(f, v_v) 
            #  - p_ext* ufl.inner(ufl.grad(self.g), v_v)\
            # + ufl.inner(self.g,ufl.div(p_ext*v_v))
              ) * ufl.dx \
            + self.g * p_ext * ufl.inner(n, v_v) * ufl.ds \
            + self.g*η*ufl.inner(mf.ε(du_v), mf.ε(v)) * ufl.dx \
            - ufl.inner(self.p, ufl.div(v)) * ufl.dx \
            - ufl.inner(σ, mf.ε(v)) * ufl.dx \
            - ufl.inner(ufl.div(du_v), q) * ufl.dx \
            

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

        
        



        

    def setup_damage(self):

        C3 = self.params.C3; l = self.params.lstar;
        ψcrit = self.params.ψcritstar; ν = self.params.ν

    
        H = mf.history_function(mf.ε(self.u_e),self.Hprev,ν,ψcrit,
                                    self.free_energy_plus)


        

        dissipated_energy = (1/C3) * mf.crack_density_function(self.d,l)*ufl.dx
        elastic_energy = self.g * H * ufl.dx
       
        total_energy = dissipated_energy + elastic_energy #- self.external_energy_without_surface()



        F = ufl.derivative(total_energy,self.d,ufl.TestFunction(self.D))
        J = ufl.derivative(F,self.d,ufl.TrialFunction(self.D))

        self.damage_problem = solvers.SNESProblem(F, self.d, bcs=self.bc_d)

        self.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)

        self.damage_solver.setFunction(self.damage_problem.F, fem.petsc.create_vector(fem.form(F)))
        self.damage_solver.setJacobian(self.damage_problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)


        
        self.damage_solver.setType("newtonls")

        
        
        self.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.damage_solver.getKSP().setType("preonly")
        self.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
        self.damage_solver.getKSP().getPC().setType("lu")

    def update_history(self):

    


        h, g = ufl.TrialFunction(self.H_space), ufl.TestFunction(self.H_space)

        H = mf.history_function(mf.ε(self.u_e),self.Hprev,
                                self.params.ν,self.params.ψcritstar)

        a = ufl.inner(h,g) * ufl.dx
        L = ufl.inner(H,g) * ufl.dx

        problem = fem.petsc.LinearProblem(a, L, 
                [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        
        self.Hprev = problem.solve()

        # interpolate the history function
        # self.Hprev.interpolate(fem.Expression(H,self.H_space.element.interpolation_points()))

    def solve_displacement(self):
        self.solver.solve(None, self.w.x.petsc_vec)

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def fixed_point(self, max_its=100, tol=1e-4, solve_stokes=False):
        L2_old = 0.0

        one = fem.Function(self.D)
        one.x.array[:] = 1.0
        area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

        area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))


        
        for i in range(max_its):
            
            self.solve_damage()
            self.solve_displacement()
            if solve_stokes:
                self.solve_velocity()
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)/area
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")

            if i>0:
                if error_L2 < tol:
                    break

            L2_old = L2

        # Update history function as finished fixed point iteration
        self.update_history()
        


    

    
    def lagrangian_update(self):
        
        uhh = fem.Function(self.V)
        uhh.interpolate(fem.Expression(self.u - self.u_prev_time,self.V.element.interpolation_points()))
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.uc_star*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        # self.w.x.array[:] = 0.0
        
