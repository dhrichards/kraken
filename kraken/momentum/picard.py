import adios4dolfinx
from .base import Momentum
import numpy as np
from dolfinx import fem, mesh
import dolfinx
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc
from kraken.numerics.invariants import matrix_function
from kraken.boundaryconditions import marked_ds
from time import perf_counter


class Picard(Momentum):
    '''
    Class for solving the momentum equation for a Maxwell viscoelastic material

    Time evolution is handled using a semi-Lagrangian approach, such that it solves for the change
    in displacements at each timestep and the mesh is then moved. 

    Solves for the change in total displacement, change in viscous displacement, and pressure as a mixed function.
    The elastic displacement is then calculated as the difference between the total and viscous displacements.
    The elastic strain is then calculated from the elastic displacement and the previous timestep's elastic strain.

    The viscosity can be non-newtonian, for some power law
    '''

    def __init__(self, sim):
        super().__init__(sim)

        self.elastic_time =0.0
        self.viscous_time = 0.0

        self.mesh_smoothing = False

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.p_el])

        self.U = fem.functionspace(self.sim.msh, self.u_el)
        self.W = fem.functionspace(self.sim.msh, self.mixed_el)


        self.du_e = fem.Function(self.U, name="change in total displacement")
        self.w = fem.Function(self.W, name="mixed function")

        self.du_v, self.dp = ufl.split(self.w)
        # self.du_e = self.du - self.du_v
        self.du = self.du_v + self.du_e
        # self.w.x.array[:] =1.0


        self.du_e_prev_it = fem.Function(self.U, name="total displacement prev it")
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")

        self.du_v_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        # self.du_e_prev_it = self.du_prev_it - self.du_v_prev_it
        self.du_prev_it = self.du_e_prev_it + self.du_v_prev_it

        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_e_prev_time = fem.Function(self.U, name="total displacement prev time")

        self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        # self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time
        self.u_prev_time = self.u_v_prev_time + self.u_e_prev_time
        
        self.bc_e = self.sim.bc_funcs[0][0](self.U)
        self.bc_s = self.sim.bc_funcs[0][1](self.W)

        

        

        self.u = self.u_prev_time + self.du
        self.u_v = self.u_v_prev_time + self.du_v
        self.u_e = self.u_e_prev_time + self.du_e
        self.p =  self.p_prev_time + self.dp

        self.u_prev_it = self.u_prev_time + self.du_prev_it
        self.u_v_prev_it = self.u_v_prev_time + self.du_v_prev_it
        self.u_e_prev_it = self.u_e_prev_time + self.du_e_prev_it
        self.p_prev_it = self.p_prev_time + self.dp_prev_it

        self.vel = self.du_v/self.sim.params.dtstar
        self.vel_prev_it = self.du_v_prev_it/self.sim.params.dtstar

        self.pw = self.water_pressure(self.du)
        self.p_crack = self.crack_pressure()


        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsiloneprevtime")
        self.ε_e = mf.ε(self.du_e) + self.ε_e_prev_time
        self.ε_e_prev_it = mf.ε(self.du_e_prev_it) + self.ε_e_prev_time

        self.ψplus = self.free_energy_plus(self.ε_e)

    

        

        


    def setup_momentum(self):
        self.setup_elastic()
        self.setup_stokes()


    def setup_elastic(self):


        v = ufl.TestFunction(self.U)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)
        

        # σ0 = es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν)
        σ = self.stress(self.ε_e,self.ε_e_prev_it)

        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)


        self.F_e = (
            + ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
              ) * ufl.dx 


        if self.sim.basal_friction:
            # water pressure on other boundaries
            self.F_e += (
                self.pw * ufl.inner(n, v) * self.sim.marked_ds(2)\
                )

            # linear weertman on bottom boundary
            self.F_e += (
                self.sim.params.βstar*ufl.inner(self.du/self.sim.params.dtstar,v)*self.sim.marked_ds(1)
            )
        else:
            self.F_e += (
                self.pw * ufl.inner(n, v) * ufl.ds\
                )
        
        self.J_e = ufl.derivative(self.F_e,self.du_e,ufl.TrialFunction(self.U))
        
        self.problem_e = solvers.SNESProblem(self.F_e, self.du_e, bcs=self.bc_e)

    def setup_stokes(self):

            w_test = ufl.TestFunction(self.W)
            v_v, q = ufl.split(w_test)

    
            g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)
            
            σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
            
            if self.sim.params.n.value==1.0:
                η0 = 1.0
            else:
                η0 = mf.viscosity(ufl.dev(mf.ε(self.vel_prev_it)), self.sim.params.n, 1e-19)
            η = (1-self.sim.damage.d)**2*η0 + self.sim.params.viscosity_tol
        
    
            self.F_s = (
                    # η0*ufl.inner(εD, mf.ε(v_v))\
                    η*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
                    - g*ufl.inner(self.p, ufl.div(v_v))  \
                -    g*ufl.inner(σ0, mf.ε(v_v))
                    ) * ufl.dx
            
    
            self.F_s += (
                    - g*ufl.div(self.du_v)*q \
                    ) * ufl.dx 
        
    
            self.J_s = ufl.derivative(self.F_s,self.w,ufl.TrialFunction(self.W))
            
            self.problem_s = solvers.SNESProblem(self.F_s, self.w, bcs=self.bc_s)
            
    

        
    def setup_solver(self):
    
    
            self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
            self.solver_s = PETSc.SNES().create(MPI.COMM_WORLD)

            
            for solver in [self.solver,self.solver_s]:
                solver.setTolerances(rtol=1.0e-11, max_it=10, atol=1e-13)
                solver.getKSP().setType("preonly")
                # solver.getKSP().setTolerances(rtol=1.0e-7)
                solver.getKSP().getPC().setType("lu")
                solver.getKSP().getPC().setFactorSolverType("mumps")
     
            # self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.extract_function_spaces(fem.form(self.F))))
            self.solver.setFunction(self.problem_e.F,dolfinx.fem.petsc.create_vector(fem.form(self.F_e)))
            self.solver.setJacobian(self.problem_e.J,dolfinx.fem.petsc.create_matrix(fem.form(self.J_e)),P=None)

            self.solver_s.setFunction(self.problem_s.F,dolfinx.fem.petsc.create_vector(fem.form(self.F_s)))
            self.solver_s.setJacobian(self.problem_s.J,dolfinx.fem.petsc.create_matrix(fem.form(self.J_s)),P=None)            


    def solve(self):

        comm = MPI.COMM_WORLD

        comm.Barrier()
        t0 = perf_counter()

        self.solver.solve(None,self.du_e.x.petsc_vec)
        self.du_e.x.scatter_forward()

        comm.Barrier()
        t1 = perf_counter()

        self.solver_s.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()

        comm.Barrier()
        t2 = perf_counter()

        self.w_prev_it.x.array[:] = self.w.x.array[:]
        self.du_e_prev_it.x.array[:] = self.du_e.x.array[:]

        self.elastic_time += t1 - t0
        self.viscous_time += t2 - t1

    def timestep(self):

        self.ε_e_prev_time.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))


        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))

        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]
        self.u_e_prev_time.x.array[:] += self.du_e.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        self.elastic_time = 0
        self.viscous_time = 0
        

        
    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

        
        
    