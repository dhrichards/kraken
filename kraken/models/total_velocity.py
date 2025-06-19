import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import numpy as np
from kraken.models import surface
from kraken.numerics import maths_functions as mf
from kraken.numerics import total_velocity_maths as mt
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc

class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, params, g=mf.degradation_default):
        self.msh = msh
        self.params = params
        self.free_energy_plus = es.free_energy_plus_spectral

        self.bounded = False
        self.w = lambda d: d**2 # dissipation function
        
        self.U = fem.functionspace(self.msh, ("Lagrange", 2, (self.msh.geometry.dim,)))
        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim,)))
        self.Q = fem.functionspace(self.msh, ("Lagrange", 1))
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))
        self.T = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, self.msh.geometry.dim)))
        self.H_space = fem.functionspace(self.msh, ("DG", 0))

        self.bc_u = bc_funcs[0](self.U)
        self.bc_d = bc_funcs[1](self.D)
        
        self.u = fem.Function(self.U, name="velocity")
        self.u.x.array[:] = 1.0
        self.p = fem.Function(self.Q, name="pressure")
        self.u_prev_it = fem.Function(self.U, name="velocity_prev")
        self.p_prev_time = fem.Function(self.Q, name="pressure_prev")
        self.σD_prev_time = fem.Function(self.T, name="stress_prev")

        self.d = fem.Function(self.D, name="damage")
        self.d_lb = fem.Function(self.D, name="damage_lb")
        self.d_ub = fem.Function(self.D, name="damage_ub")

        self.d_lb.x.array[:] = 0.0
        self.d_ub.x.array[:] = 1.0
        self.Hprev = fem.Function(self.H_space, name="history")

        self.g = g(self.d)
        
        

        self.move_mesh = self.lagrangian_update
      
        

    def setup_damage(self):

        C3 = self.params.C3; l = self.params.lstar
        ψcrit = self.params.ψcritstar; ν = self.params.ν
        
        s = np.linspace(0,1,500)
        self.c0 = 4*np.trapezoid(np.sqrt(self.w(s)),s)

        if self.bounded:
            H = self.free_energy_plus(mt.elastic_strain(self.σD,self.p,ν),ν) - ψcrit
        else:
            H = mf.history_function(mt.elastic_strain(self.σD,self.p,ν),self.Hprev,ν,ψcrit,
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

        
        
        self.damage_solver.setTolerances(rtol=1.0e-12, max_it=50)
        self.damage_solver.getKSP().setType("preonly")
        self.damage_solver.getKSP().setTolerances(rtol=1.0e-12)
        self.damage_solver.getKSP().getPC().setType("lu")

    def update_history(self):

        if self.bounded:
            self.d_lb.x.array[:] = self.d.x.array[:]

        else:

            ν = self.params.ν

            h, g = ufl.TrialFunction(self.H_space), ufl.TestFunction(self.H_space)

            H = mf.history_function(mt.elastic_strain(self.σD,self.p,ν),self.Hprev,
                                    ν,self.params.ψcritstar)

            a = ufl.inner(h,g) * ufl.dx
            L = ufl.inner(H,g) * ufl.dx

            problem = fem.petsc.LinearProblem(a, L, 
                    [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            
            self.Hprev = problem.solve()

    def update_stress(self):
        
        σD, τ = ufl.TrialFunction(self.T), ufl.TestFunction(self.T)

        
        a = ufl.inner(σD, τ) * ufl.dx
        L = ufl.inner(self.σD, τ) * ufl.dx
        problem = fem.petsc.LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        self.σD_prev_time = problem.solve()

    def setup_velocity(self):

        De = self.params.De
        # De = 0.0
        λoverμ = self.params.λ/self.params.μ
        D = self.msh.geometry.dim


        du, dp = ufl.TrialFunction(self.U), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.U), ufl.TestFunction(self.Q)

        self.n = ufl.FacetNormal(self.msh)
        self.ds = ufl.Measure("ds", domain=self.msh)
        
        self.p_ext = mf.water_pressure(self.msh,self.u,self.params.ucstar) +self.params.patmstar
        self.f = self.g*mf.body_force(self.msh, self.params.ρistar, self.params.slope_angle)
        
        self.η = mf.viscosity(mf.εD(self.u), self.params.n)
        # self.η = 1.0
        self.σD = mt.deviatoric_stress(mf.εD(self.u), self.σD_prev_time, self.η, De)
        self.η_mod = self.η/(1 + De*self.η)

        self.p_mod = 2*self.η_mod * De/ (D*(λoverμ + 2/D))
        
        
        F = [(self.g*2*self.η_mod*ufl.inner(mf.ε(self.u), mf.ε(v)) \
        - ufl.inner(self.p*(1-self.p_mod), ufl.div(v)) \
        + ufl.inner(self.p_mod*self.p_prev_time, ufl.div(v)) \
        - ufl.inner(De*self.η/(1+De*self.η)*self.σD_prev_time, mf.ε(v)) \
        - ufl.inner(self.f, v) 
        - self.p_ext* ufl.inner(ufl.grad(self.g), v)\
            ) * ufl.dx \
        + self.g * self.p_ext * ufl.inner(self.n, v) * self.ds \
        ,
        - (ufl.inner(ufl.div(self.u), q) - (De/(D*(λoverμ + 2/D)))*(self.p-self.p_prev_time)*q)* ufl.dx ]

        # F = [(self.g*2*self.η*ufl.inner(mf.ε(self.u), mf.ε(v)) \
        # - ufl.inner(self.p, ufl.div(v)) \
        # - ufl.inner(self.f, v) \
        # - self.p_ext * ufl.inner(ufl.grad(self.g), v)\
        #     ) * ufl.dx \
        # + self.g * self.p_ext* ufl.inner(self.n, v) * self.ds,
        # - ufl.inner(ufl.div(self.u), q) * ufl.dx ]
        
        J = [[ufl.derivative(F[0], self.u, du), ufl.derivative(F[0], self.p, dp)],
            [ufl.derivative(F[1], self.u, du), ufl.derivative(F[1], self.p, dp)]]
        
        P = [[J[0][0], None],
            [None, (2 * self.g*self.η_mod)**-1 * dp * q * ufl.dx]]
        

        self.stokes_solver, self.x = solvers.nested_solve(F, J, self.u, self.p, self.bc_u, P)

        opts = PETSc.Options()
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"
        
        # opts["snes_rtol"] = 1.0e-7
        self.stokes_solver.setFromOptions()

        

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def solve_velocity(self):
        # self.stokes.solve(self.u, self.p, self.d, self.v)
        self.stokes_solver.solve(None, self.x)

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()

        self.u_prev_it.x.array[:] = self.u.x.array[:]
        
        

    def fixed_point(self, max_its=100, tol=1e-4, solve_damage=True):
        L2_old = 0.0

        one = fem.Function(self.D)
        one.x.array[:] = 1.0
        area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

        area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))


        
        for i in range(max_its):
            
            if solve_damage: 
                self.solve_damage()
            self.solve_velocity()
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx + ufl.inner(self.u,self.u)*ufl.dx
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
        if solve_damage:
            self.update_history()
        self.update_stress()
        self.p_prev_time = fem.Function(self.Q, name="pressure_prev_time")
        self.p_prev_time.x.array[:] = self.p.x.array[:]


    def start_elastic(self):

        ν = self.params.ν

        self.v = fem.Function(self.U, name="elastic_displacement")

        elastic_energy = es.free_energy(mf.ε(self.v), ν) * ufl.dx

        uestar = self.params.ucstar*self.params.De

        p_ext = mf.water_pressure(self.msh, self.v, uestar) + self.params.patmstar

        external_energy = (ufl.dot(self.f, self.v) 
                           + p_ext * ufl.inner(ufl.grad(self.g), self.v)\
                            ) * ufl.dx \
                            - p_ext * ufl.dot(self.n,self.v) * self.ds
        
        total_energy = elastic_energy - external_energy

        F = ufl.derivative(total_energy, self.v, ufl.TestFunction(self.U))
        J = ufl.derivative(F, self.v, ufl.TrialFunction(self.U))

        elastic_problem = solvers.SNESProblem(F, self.v, bcs=self.bc_u)
        elastic_solver = PETSc.SNES().create(MPI.COMM_WORLD)

        elastic_solver.setTolerances(rtol=1.0e-7, max_it=50)
        elastic_solver.getKSP().setType("preonly")
        elastic_solver.getKSP().setTolerances(rtol=1.0e-7)
        elastic_solver.getKSP().getPC().setType("lu")

        elastic_solver.setFunction(elastic_problem.F, fem.petsc.create_vector(fem.form(F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        elastic_solver.setJacobian(elastic_problem.J, fem.petsc.create_matrix(fem.form(J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        elastic_solver.solve(None, self.v.x.petsc_vec)

        λoverμ = self.params.λ/self.params.μ
        D = self.msh.geometry.dim

        self.p_prev_time = -(λoverμ + 2/D)*ufl.div(self.v)
        self.σD_prev_time = 2*mf.εD(self.v)
                

    
    def lagrangian_update(self):
        
        uhh = fem.Function(self.V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.ucstar*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        self.u.x.array[:] = 0.0
        self.u_prev_it.x.array[:] = 0.0



