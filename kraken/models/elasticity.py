import numpy as np
from dolfinx import fem, default_scalar_type
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



class elastic_damage:
    def __init__(self, msh, bc_funcs):
        self.msh = msh
        self.params = parameters.Params_with_uc(self.msh)

        self.u_el = bufl.element("CG", self.msh.basix_cell(), 1, shape=(self.msh.geometry.dim,))
        
        self.U = fem.functionspace(self.msh, self.u_el)

        self.u = fem.Function(self.U, name="displacement")
        self.ε_e = mf.ε(self.u)

        self.u_prev_it = fem.Function(self.U, name="displacement previous iteration")
        self.u_prev_time = fem.Function(self.U, name="displacement previous time")
 
        self.D = fem.functionspace(self.msh, ("Lagrange", 1))

        self.H_el = bufl.quadrature_element(
            self.msh.basix_cell(), value_shape=(), scheme="default", degree=1
        )
        self.H_space = fem.functionspace(self.msh, ("DG", 1))



        self.bc_u = bc_funcs[0](self.U)
        self.bc_d = bc_funcs[1](self.D)

        # damage.setup_higher_order_spaces(self,bc_funcs[1])
        self.d = fem.Function(self.D, name="damage")
        self.g = mf.degradation_Lo2023(self.d, 5)
        self.g = es.degradation_default(self.d)
        self.d_prev_time = fem.Function(self.D, name="damage previous time")
        self.Hprev = fem.Function(self.H_space, name="history")

      
     
    def setup_all(self):
        self.setup_momentum()
        # self.setup_damage()
        # damage.setup_damage_non_linear(self)
        damage.setup_damage_bounded(self, free_energy_plus=es.free_energy_plus_lo)
        # damage.setup_damage_higher_order(self,es.free_energy_plus_dp)


    def setup_momentum(self):


        v = ufl.TestFunction(self.U)

        p_w = mf.water_pressure(self.msh,self.u,self.params.ucstar) +self.params.patmstar
        p_crack = mf.water_pressure_static(self.msh, level=0.05) 

        
        
        σ0 = es.cauchy_stress(self.ε_e, self.params.ν)
        σplus = es.stress_plus_lo(self.ε_e, self.params.ν)
        σminus = σ0 - σplus
        σ = self.g*σplus + σminus

        f = mf.body_force(self.msh, self.params.ρistar)



        n = ufl.FacetNormal(self.msh)

        F = (ufl.inner(σ, mf.ε(v))\
            #  - (1-g)*ufl.inner(p_ext, ufl.div(v_v))
              - ufl.inner(f, v) 
             - p_crack* ufl.inner(ufl.grad(self.g), v)\
            # - mf.overburden_pressure(self.msh, self.params.ρistar, self.u, self.params.ucstar)*ufl.inner(ufl.grad(g), v_v)
              ) * ufl.dx \
            + p_w * ufl.inner(n, v) * ufl.ds 
        

        J = ufl.derivative(F,self.u,ufl.TrialFunction(self.U))
            
        
        self.problem = solvers.SNESProblem(F, self.u, bcs=self.bc_u)

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
      

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
        self.solver.solve(None, self.u.x.petsc_vec)
        # self.w.x.scatter_forward()
        self.u_prev_it.x.array[:] = self.u.x.array[:]

    def solve_damage(self):
        self.damage_solver.solve(None, self.d.x.petsc_vec)


    def timestep(self):
        self.u_prev_time.x.array[:] = self.u.x.array[:]
        self.update_history()
        # self.d_prev_time.x.array[:] = self.d.x.array[:]
        # self.d_prev_time.x.array[:] = self.d.x.array[:]
        # self.w.x.array[:] = 0.0
