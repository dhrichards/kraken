import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import numpy as np
from kraken.models import surface
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc

class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, material, dt, g=mf.degradation_default, eulerian_surfaces=False,acc=lambda x: 0.0):
        self.msh = msh
        self.material = material
        self.dt = dt

        self.bounded = False
        self.w = lambda d: d**2 # dissipation
        self.free_energy_plus = es.free_energy_plus_spectral

        

        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim,)))
        self.U = fem.functionspace(self.msh, ("Lagrange", 2, (self.msh.geometry.dim,)))
        self.Q = fem.functionspace(self.msh, ("Lagrange", 1))
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))
        self.H_space = fem.functionspace(self.msh, ("DG", 0))

        self.bc_v = bc_funcs[0](self.V)
        self.bc_u = bc_funcs[1](self.U)
        self.bc_d = bc_funcs[2](self.D)

        self.v = fem.Function(self.V, name="elastic displacement")
        
        self.u = fem.Function(self.U, name="velocity")
        self.p = fem.Function(self.Q, name="pressure")
        self.u_prev = fem.Function(self.U, name="velocity_prev")

        self.d = fem.Function(self.D, name="damage")
        self.d_lb = fem.Function(self.D, name="damage_lb")
        self.d_ub = fem.Function(self.D, name="damage_ub")
        self.d_prev = fem.Function(self.D, name="damage_prev")

        self.d_lb.x.array[:] = 0.0
        self.d_ub.x.array[:] = 1.0
        self.Hprev = fem.Function(self.H_space, name="history")

        self.g = g(self.d)
        self.p_ext = lambda u: mf.water_pressure(self.msh,u) +self.material.patmstar
        self.η = mf.viscosity(self.u, self.material.n, 1.e-8)
        self.f = self.g*mf.body_force(self.msh, self.material.ρratio, self.material.slope_angle)
        self.n = ufl.FacetNormal(self.msh)
        self.ds = ufl.Measure("ds", domain=self.msh)
        
        self.Iprime = 2*self.d # derivative of Indicator function, see https://doi.org/10.1016/j.tafmec.2023.104040
        # self.Iprime = 6*self.d - 6*self.d**2

     
        


        if eulerian_surfaces:
            self.surface = surface.SurfaceSolver(self.msh, bc_funcs[3], self.material, self.dt, eulerian_surfaces, acc)
            self.z = fem.Function(self.surface.V, name="z")
            self.z.interpolate(lambda x: x[self.msh.geometry.dim-1])
            self.move_mesh = self.eulerian_update
        else:
            self.move_mesh = self.lagrangian_update



    def setup_all(self):
        self.setup_elastic()
        self.setup_damage()
        self.setup_stokes()

    def setup_elastic(self):

        ν = self.material.ν; C1 = self.material.C1; ψcrit = self.material.ψcritstar

        elastic_energy = mf.degraded_free_energy(mf.ε(self.v), self.g, ν,
                                        ψcrit, self.free_energy_plus) * ufl.dx
        
        external_energy = ( C1*ufl.dot(self.f, self.v) \
            + C1*self.p_ext(self.v)*ufl.inner(ufl.grad(self.g), self.v)\
            # - self.pw*self.Iprime*ufl.inner(ufl.grad(self.d), self.v)\
            )* ufl.dx \
            - C1*self.g*self.p_ext(self.v)*ufl.dot(self.n, self.v) * self.ds
        

        total_energy = elastic_energy - external_energy

        F = ufl.derivative(total_energy,self.v,ufl.TestFunction(self.V))
        J = ufl.derivative(F,self.v,ufl.TrialFunction(self.V))
        
        self.elastic_problem = solvers.SNESProblem(F, self.v, bcs=self.bc_v)

        self.elastic_solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setType("newtonls")
        # opts = PETSc.Options()
        # opts["snes_type"] = "newtonls"
        # opts["snes_linesearch_type"] = "bt"

        # self.elastic_solver.setFromOptions()

        self.elastic_solver.setTolerances(rtol=1.0e-7, max_it=50)
        self.elastic_solver.getKSP().setType("preonly")
        self.elastic_solver.getKSP().setTolerances(rtol=1.0e-7)
        self.elastic_solver.getKSP().getPC().setType("lu")
        # self.solver.getKSP().getPC().setFactorSolverType("mumps")
 

        self.elastic_solver.setFunction(self.elastic_problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        self.elastic_solver.setJacobian(self.elastic_problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        
        

    def setup_damage(self):

        C3 = self.material.C3; l = self.material.lstar;
        ψcrit = self.material.ψcritstar; ν = self.material.ν
        
        s = np.linspace(0,1,500)
        self.c0 = 4*np.trapz(np.sqrt(self.w(s)),s)

        if self.bounded:
            H = self.free_energy_plus(mf.ε(self.v),ν) - ψcrit
        else:
            H = mf.history_function(mf.ε(self.v),self.Hprev,ν,ψcrit,
                                    self.free_energy_plus)


        

        dissipated_energy = (1/C3) * mf.crack_density_function(self.d,l,self.w, self.c0)*ufl.dx
        elastic_energy = self.g * H * ufl.dx
       
        total_energy = dissipated_energy + elastic_energy #- self.external_energy_without_surface()



        F = ufl.derivative(total_energy,self.d,ufl.TestFunction(self.D))
        J = ufl.derivative(F,self.d,ufl.TrialFunction(self.D))

        self.damage_problem = solvers.SNESProblem(F, self.d, bcs=self.bc_d)

        self.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)

        self.damage_solver.setFunction(self.damage_problem.F, fem.petsc.create_vector(fem.form(F)))
        self.damage_solver.setJacobian(self.damage_problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)


        if self.bounded:
            self.damage_solver.setType("vinewtonrsls")
            self.damage_solver.setVariableBounds(self.d_lb.x.petsc_vec,self.d_ub.x.petsc_vec)
        else:
            self.damage_solver.setType("newtonls")

        
        
        self.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.damage_solver.getKSP().setType("preonly")
        self.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
        self.damage_solver.getKSP().getPC().setType("lu")

    def update_history(self):

        if self.bounded:
            self.d_lb.x.array[:] = self.d.x.array[:]

        else:



            h, g = ufl.TrialFunction(self.H_space), ufl.TestFunction(self.H_space)

            H = mf.history_function(mf.ε(self.v),self.Hprev,
                                    self.material.ν,self.material.ψcritstar)

            a = ufl.inner(h,g) * ufl.dx
            L = ufl.inner(H,g) * ufl.dx

            problem = fem.petsc.LinearProblem(a, L, 
                    [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            
            self.Hprev = problem.solve()


    def setup_stokes(self):

        du, dp = ufl.TrialFunction(self.U), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.U), ufl.TestFunction(self.Q)

        C1 = self.material.C1; C2 = self.material.C2

    
        
        
        F = [((1/C2)*self.g*2*self.η*ufl.inner(mf.ε(self.u), mf.ε(v)) \
        - ufl.inner(self.p, ufl.div(v)) \
        - C1 * ufl.inner(self.f, v) \
        - C1 * self.p_ext(self.u*self.dt) * ufl.inner(ufl.grad(self.g), v)\
            ) * ufl.dx \
        + C1 * self.g * self.p_ext(self.u*self.dt) * ufl.inner(self.n, v) * self.ds,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]
        
        J = [[ufl.derivative(F[0], self.u, du), ufl.derivative(F[0], self.p, dp)],
            [ufl.derivative(F[1], self.u, du), ufl.derivative(F[1], self.p, dp)]]
        
        P = [[J[0][0], None],
            [None, (2 * self.g*self.η)**-1 * dp * q * ufl.dx]]
        

        self.stokes_solver, self.x = solvers.nested_solve(F, J, self.u, self.p, self.bc_u, P)

        opts = PETSc.Options()
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"
        
        # opts["snes_rtol"] = 1.0e-7
        self.stokes_solver.setFromOptions()

        


    def solve_elastic(self):
        self.elastic_solver.solve(None, self.v.x.petsc_vec)

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def solve_stokes(self):
        # self.stokes.solve(self.u, self.p, self.d, self.v)
        self.stokes_solver.solve(None, self.x)

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
        self.u_prev.x.array[:] = self.u.x.array[:]

    def fixed_point_simple(self, max_its=100, tol=1e-4, solve_stokes=False):
        L2_old = 0.0

        one = fem.Function(self.D)
        one.x.array[:] = 1.0
        area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

        area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))


        
        for i in range(max_its):
            
            self.solve_damage()
            self.solve_elastic()
            if solve_stokes:
                self.solve_stokes()
            

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
        


    
    def fixed_point(self, max_its=100, tol=1e-4, solve_stokes=False):
        L2_old = 0.0

        self.converged = False
        d_old = self.d.copy()
        v_old = self.v.copy()

        for i in range(max_its):

            self.solve_elastic()
            self.solve_damage()
            if solve_stokes:
                # self.stokes.solve_linearised(self.u,self.p,self.d,self.v)
                self.stokes.solve()
            # self.solver_d.solve(None, self.d.x.petsc_vec)
            # self.solve_damage_limits()
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")

            if error_L2 > 100.0: # this is a test for the whole region being damaged
                break
            
            if i>0:
                if error_L2 < tol:
                    self.converged = True
                    break

            L2_old = L2

        # Update history function as finished fixed point iteration
        if self.converged:
            self.update_history()


                

    
    def lagrangian_update(self):
        
        uhh = fem.Function(self.V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.dt*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        self.u.x.array[:] = 0.0
        self.u_prev.x.array[:] = 0.0


    def eulerian_update(self):
        self.z = self.surface.solve_nitshe(self.u,self.z)

        self.msh.geometry.x[:,self.msh.geometry.dim-1] = self.z.x.array


