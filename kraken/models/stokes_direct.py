import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl
import numpy as np
import basix.ufl as bufl
from kraken.models import damage
from kraken.numerics import maths_functions as mf
from kraken.numerics import total_velocity_maths as mt
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc

class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, params,):
        self.msh = msh
        self.params = params

        self.u_el = bufl.element("CG", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.p_el])

        self.W = fem.functionspace(self.msh, self.mixed_el)
        self.w = fem.Function(self.W, name="mixed function")

        self.u, self.p = ufl.split(self.w)

        self.D = fem.functionspace(self.msh, ("Lagrange", 1))
        self.H_space = fem.functionspace(self.msh, ("DG", 1))
        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim,)))

        self.bc_u = bc_funcs[0](self.W)
        self.bc_d = bc_funcs[1](self.D)
        
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.u_prev_it, self.p_prev_it = ufl.split(self.w_prev_it)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)



        self.d = fem.Function(self.D, name="damage")
        self.d_prev_time = fem.Function(self.D, name="damage_prev_time")
        self.Hprev = fem.Function(self.H_space, name="history")

        self.g = mf.degradation_default(self.d)
        
        

    def update_history(self):

        H = mf.history_function(self.ε_e,self.Hprev,
                                self.params.ν,self.params.ψcritstar)

        self.Hprev.interpolate(fem.Expression(H,self.H_space.element.interpolation_points()))


    def setup_all(self,compressible=True):
        self.setup(compressible)
        damage.setup_damage_bounded(self)

    def setup(self,compressible=True):

        δt = self.params.dtstar
        λoverμ = self.params.λ/self.params.μ
        D = self.msh.geometry.dim
        κ = λoverμ + 2/D

        w_test = ufl.TestFunction(self.W)
        v, q = ufl.split(w_test)

        dot_u = (self.u - self.u_prev_time)/self.params.dtstar
        dot_u_prev_it = (self.u_prev_it - self.u_prev_time)/self.params.dtstar
        
       
        n = ufl.FacetNormal(self.msh)
        
        p_ext = mf.water_pressure(self.msh,self.u,self.params.ucstar) +self.params.patmstar
        # p_ext = mf.water_pressure_static(self.msh) + self.params.patmstar
        f = self.g*mf.body_force(self.msh, self.params.ρistar, self.params.slope_angle)
        
        self.η = mf.viscosity(mf.εD(dot_u_prev_it), self.params.n)

        
        self.ε_e = self.η*mf.εD(dot_u) - self.p/(D*self.g*κ) * ufl.Identity(D)

        dot_p = (self.p_prev_it - self.p_prev_time)/self.params.dtstar

        F = (self.g*2*self.η*ufl.inner(mf.ε(dot_u), mf.ε(v)) \
        - ufl.inner(self.p \
                    # + 2*self.η*dot_p/(D*κ)\
                        , ufl.div(v)) \
        - ufl.inner(f, v) \
        - p_ext * ufl.inner(ufl.grad(self.g), v)\
            ) * ufl.dx \
        + self.g * p_ext* ufl.inner(n, v) * ufl.ds \
        + (ufl.inner(ufl.div(self.u), q) \
           +self.p*q/(self.g*κ)\
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


        

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)

    def solve(self):
        # self.stokes.solve(self.u, self.p, self.d, self.v)
        self.solver.solve(None, self.w.x.petsc_vec)

        self.w_prev_it.x.array[:] = self.w.x.array[:]
        
  
    
    def timestep(self):


        # du = self.u - self.u_prev_time
        # uhh = fem.Function(self.V)
        # uhh.interpolate(fem.Expression(du, self.V.element.interpolation_points()))
        # self.msh.geometry.x[:,:self.msh.geometry.dim] += self.params.ucstar*uhh.x.array.reshape((-1, self.msh.geometry.dim))
        
        self.d_prev_time.x.array[:] = self.d.x.array[:]
        self.w_prev_time.x.array[:] = self.w.x.array[:]

