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
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc
from kraken.numerics.invariants import matrix_function
from kraken.boundaryconditions import marked_ds


class SemiLagrangianEpsilon(Momentum):

    def __init__(self, sim):
        super().__init__(sim)

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

        self.accel = (self.u - 2*self.u_prev_time + self.u_prev_2)/(self.sim.params.dtstar**2)

        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsiloneprevtime")
        self.ε_e = mf.ε(self.du_e) + self.ε_e_prev_time
        self.ε_e_prev_it = mf.ε(self.du_e_prev_it) + self.ε_e_prev_time

        self.ε_eD = self.ε_e - (1/3)*1.5*ufl.tr(self.ε_e)*ufl.Identity(self.sim.msh.geometry.dim)

        self.ψplus = self.free_energy_plus(self.ε_e,self.du)

        

        


    def setup_momentum(self):
        w_test = ufl.TestFunction(self.W)
        v, v_v, q = ufl.split(w_test)
        n = ufl.FacetNormal(self.sim.msh)

        g = es.degradation_default(self.sim.damage.d,self.sim.params.ge_tol)
        

        # σ0 = es.cauchy_stress(self.ε_e_prev_it,self.sim.params.ν)
        σ = self.stress(self.ε_e,self.du)
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        
        A = mf.rate_factor(self.sim.params.T)/self.sim.params.A0

        η0 = mf.viscosity(ufl.dev(mf.ε(self.vel_prev_it)), self.sim.params.n, 1e-13, A=A)

        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)

        Iprime = 2*self.sim.damage.d

        def right_boundary(x):
            return np.isclose(x[0], self.sim.params.length.value/self.sim.params.H.value)
        
        def bottom_boundary(x):
            return np.isclose(x[1], 0.0)
        
        def left_boundary(x):
            return np.isclose(x[0], 0.0)
        
        
        r_facets = mesh.locate_entities_boundary(self.sim.msh, self.sim.msh.topology.dim-1, right_boundary)
        b_facets = mesh.locate_entities_boundary(self.sim.msh, self.sim.msh.topology.dim-1, bottom_boundary)
        l_facets = mesh.locate_entities_boundary(self.sim.msh, self.sim.msh.topology.dim-1, left_boundary)
        facets = np.hstack([r_facets, b_facets, l_facets])
        values = np.hstack([np.full_like(r_facets, 1), np.full_like(b_facets, 2), np.full_like(l_facets, 3)])
        sorted_facets = np.argsort(facets)
        mt = mesh.meshtags(self.sim.msh, self.sim.msh.topology.dim-1, facets[sorted_facets], values[sorted_facets])
        ds = ufl.Measure("ds", domain=self.sim.msh, subdomain_data=mt)

        x = ufl.SpatialCoordinate(self.sim.msh)
        δ = 0.1
        σxx_ssa = δ/2 + (x[1]-1)
        t = ufl.as_vector((σxx_ssa, 0))

        
        self.F = (
            # 0.5*self.sim.params.C_inertia*ufl.inner(self.accel, v)  \
            + ufl.inner(σ, mf.ε(v)) - ufl.inner(f, v) 
            #  - self.p_crack* ufl.inner(ufl.grad(g), v)\
            # - self.p_crack*ufl.inner(ufl.Dx(g,0), v[0]) \
            # + self.p_crack*Iprime*ufl.inner(ufl.Dx(self.sim.damage.d,0), v[0]) \
            # + self.p_crack*Iprime*ufl.inner(ufl.grad(self.sim.damage.d), v)
              ) * ufl.dx 
              
        self.F += (
            self.pw * ufl.inner(n, v) * ufl.ds\
            # +(self.pw-2e-3)*ufl.inner(n,v) * ds(1) \
            # + self.pw * ufl.inner(n, v) * ds(3) \
            # + self.pw * ufl.inner(n, v) * ds(2)\
        #     # + 1e5 * ufl.inner(self.vel, v) * self.ds_bottom(1)\
            )
        
        self.F+= (
                # η0*ufl.inner(εD, mf.ε(v_v))\
                2*g*η0*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
                - g*ufl.inner(self.p, ufl.div(v_v))  \
            -    g*ufl.inner(σ0, mf.ε(v_v))
             ) * ufl.dx
        
        # self.F += (
        #         2*η*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
        #         + ufl.inner(-self.p, ufl.div(v_v))  \
        #         - ufl.inner(σ, mf.ε(v_v))\
        #             ) * ufl.dx
        # self.F += (
        #         ufl.inner(σv0, mf.ε(v_v))\
        #         + ufl.inner(σ, mf.ε(v_v))\
        # )         * ufl.dx


        self.F += (
                - g*ufl.div(self.du)*q \
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


   




    def timestep(self):

        self.ε_e_prev_time.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))

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

        
        
    