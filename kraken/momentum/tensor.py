import adios4dolfinx
from .base import Momentum
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
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

        self.order = 1

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), self.order, shape=(self.sim.msh.geometry.dim,))
        self.q_el = bufl.element("DG", self.sim.msh.basix_cell(), self.order-1, shape=(3,))
        
        self.mixed_el = bufl.mixed_element([self.u_el, self.q_el])

        self.W = fem.functionspace(self.sim.msh, self.mixed_el)

        self.w = fem.Function(self.W, name="mixed function")
        # self.w.x.array[:] =1.0
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_prev_time, self.q_prev_time = ufl.split(self.w_prev_time)
        
        self.bc_e = self.sim.bc_funcs[0](self.W)

        self.du, self.dq = ufl.split(self.w)
        self.u = self.u_prev_time + self.du
        
        self.du_prev_it, self.dq_prev_it = ufl.split(self.w_prev_it)
        

        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), self.order-1, shape=(3,3))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsiloneprevtime")
        self.ε_e_updated = fem.Function(self.E)

        self.ε_v_dot = mf.viscous_strain(self.dq)/self.sim.params.dtstar
        self.ε_v_dot_prev_it = mf.viscous_strain(self.dq_prev_it)/self.sim.params.dtstar

        # self.ε_e = self.ε_e_prev_time + ufl.sym(L(self.du)) - viscous_strain(self.dq)

        W = self.sim.params.ucstar_float*ufl.skew(mf.L(self.du))
        self.ε_e = self.ε_e_prev_time + ufl.sym(mf.L(self.du)) - mf.viscous_strain(self.dq) \
            # + ufl.dot(W,self.ε_e_prev_time) - ufl.dot(self.ε_e_prev_time,W)

        W_prev_it = self.sim.params.ucstar_float*ufl.skew(mf.L(self.du_prev_it))
        self.ε_e_prev_it = self.ε_e_prev_time + ufl.sym(mf.L(self.du_prev_it)) - mf.viscous_strain(self.dq_prev_it) \
            + ufl.dot(W_prev_it,self.ε_e_prev_time) - ufl.dot(self.ε_e_prev_time,W_prev_it)


    

        

        


    def setup_momentum(self):

        w_test = ufl.TestFunction(self.W)
        v, r = ufl.split(w_test)
        R = mf.viscous_strain(r)
        n = ufl.FacetNormal(self.sim.msh)

        g = self.sim.damage.g
        

        # σ0 = es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν)
        σ = self.σ
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        
        if self.sim.params.n.value==1.0:
            η0 = 1.0
        else:
            η0 = mf.viscosity(self.ε_v_dot_prev_it, self.sim.params.n, 1e-19)
        η = (1-self.sim.damage.d)**2*η0 + self.sim.params.viscosity_tol

        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)


        self.F = (
            + ufl.inner(σ, ufl.sym(mf.L(v))) - ufl.inner(f, v) 
              ) * ufl.dx 


        if self.sim.basal_friction:
            # water pressure on other boundaries
            self.F += (
                self.pw * ufl.inner(n, v) * self.sim.marked_ds(2)\
                )

            # linear weertman on bottom boundary
            self.F += (
                self.sim.params.βstar*ufl.inner(self.du/self.sim.params.dtstar,v)*self.sim.marked_ds(1)
            )
        else:
            self.F += (
                self.pw * ufl.inner(n, v) * ufl.ds\
                )
        
        self.F+= (
                # η0*ufl.inner(εD, mf.ε(v_v))\
                η*ufl.inner(self.ε_v_dot, R)\
            -    g*ufl.inner(ufl.dev(σ0), R)
             ) * ufl.dx
        
    

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))
        
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_e)
        



    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # self.problem.solve()
        self.w.x.scatter_forward()

        self.w_prev_it.x.array[:] = self.w.x.array[:]

   

    def timestep(self):

        self.ε_e_updated.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))


        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))

        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))

        self.ε_e_prev_time.x.array[:] = self.ε_e_updated.x.array[:]
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0
        
    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

        
        
    