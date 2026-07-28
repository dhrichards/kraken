import adios4dolfinx
from .base import Momentum
import numpy as np
from dolfinx import fem, mesh
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


class SemiLagrangianEpsilon(Momentum):
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

        self.mesh_smoothing = True

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.u_el, self.p_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        # self.w.x.array[:] =1.0
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time

        self.w_start = fem.Function(self.W, name="mixed function at start of iteration")
        self.w_prev_it_start = fem.Function(self.W, name="mixed function previous iteration at start of iteration")

        self.w_prev_2 = fem.Function(self.W, name="mixed function 2 timesteps previous")
        self.u_prev_2, self.u_v_prev_2, self.p_prev_2 = ufl.split(self.w_prev_2)

        self.vel_prev_time = (self.u_v_prev_time - self.u_v_prev_2)/self.sim.params.dtstar
        
        self.bc_u = self.sim.bc_funcs[0](self.W)

        self.du, self.du_v, self.dp = ufl.split(self.w)
        self.du_e = self.du - self.du_v

        self.du_prev_it, self.du_v_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        self.du_e_prev_it = self.du_prev_it - self.du_v_prev_it

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
        self.p_crack = self.crack_pressure(self.du)


        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsiloneprevtime")
        self.ε_e = mf.ε(self.du_e) + self.ε_e_prev_time
        self.ε_e_prev_it = mf.ε(self.du_e_prev_it) + self.ε_e_prev_time

        self.ψplus = self.free_energy_plus(self.ε_e,self.du)

        

        


    def setup_momentum(self):

        if self.mesh_smoothing:
            self.setup_smoother()
        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)
        

        # σ0 = es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν)
        σ = self.stress(self.ε_e,self.du)
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        
        A = mf.rate_factor(self.sim.params.T)/self.sim.params.A0

        η0 = mf.viscosity(ufl.dev(mf.ε(self.vel_prev_it)), self.sim.params.n, 1e-19, A=A)
        η = (1-self.sim.damage.d)**2*η0 + self.sim.params.viscosity_tol

        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)


        self.F = (
            + ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
              ) * ufl.dx 
              
        self.F += (
            self.pw * ufl.inner(n, v) * ufl.ds\
            )
        
        self.F+= (
                # η0*ufl.inner(εD, mf.ε(v_v))\
                η*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
                - g*ufl.inner(self.p, ufl.div(v_v))  \
            -    g*ufl.inner(σ0, mf.ε(v_v))
             ) * ufl.dx
        

        self.F += (
                - g*ufl.div(self.du_v)*q \
                ) * ufl.dx 
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
            
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)


    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()

        self.w_prev_it.x.array[:] = self.w.x.array[:]


    def interpolate_from_parent(self, parent):
        super().interpolate_from_parent(parent)


        for i in range(3):
            self.w.sub(i).interpolate(parent.momentum.w.sub(i), cells0=self.sim.parent_cells, cells1=self.sim.cells)
            self.w_prev_time.sub(i).interpolate(parent.momentum.w_prev_time.sub(i), cells0=self.sim.parent_cells, cells1=self.sim.cells)
            self.w_prev_it.sub(i).interpolate(parent.momentum.w_prev_it.sub(i), cells0=self.sim.parent_cells, cells1=self.sim.cells)
            self.w_prev_it_start.sub(i).interpolate(parent.momentum.w_prev_it_start.sub(i), cells0=self.sim.parent_cells, cells1=self.sim.cells)
            self.w_prev_2.sub(i).interpolate(parent.momentum.w_prev_2.sub(i), cells0=self.sim.parent_cells, cells1=self.sim.cells)
  
        self.ε_e_prev_time.interpolate(parent.momentum.ε_e_prev_time, cells0=self.sim.parent_cells, cells1=self.sim.cells)


   


    def setup_smoother(self):
        g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)

        self.du_1 = fem.Function(self.V, name="du 1")

        self.du_smooth = fem.Function(self.V, name="du smooth")
        du = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = g*ufl.inner(du, v) * ufl.dx + self.sim.params.lstar**2*ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx
        L = g*ufl.inner(self.du_1, v) * ufl.dx

        self.smooth_problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type":"preonly","pc_type":"lu"})
    

    def timestep(self):

        self.ε_e_prev_time.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))

        
        if self.mesh_smoothing:
            self.du_1.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))
            du = self.smooth_problem.solve()
            self.du_smooth.x.array[:] = du.x.array[:] # for saving
        else:
            du = fem.Function(self.V)
            du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))

        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_2.x.array[:] = self.w_prev_time.x.array[:]
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0
        

        self.w_start.x.array[:] = self.w.x.array[:]
        self.w_prev_it_start.x.array[:] = self.w_prev_it.x.array[:]

    def revert(self):
        self.w.x.array[:] = self.w_start.x.array[:]
        self.w_prev_it.x.array[:] = self.w_prev_it_start.x.array[:]

        
    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

        
        
    