import numpy as np
from dolfinx import fem
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc
import adios4dolfinx

def L(du):
    # Mesh coordinates: (x, z)
    # Tensor coordinates: (x, y, z)
    return ufl.as_tensor([
        [du[0].dx(0), 0, du[0].dx(1)],
        [0,             0, 0],
        [du[1].dx(0), 0, du[1].dx(1)],
    ])

class Momentum:
    '''Base class for solving the momentum equation.'''
    def __init__(self, sim):
        self.sim = sim
        

        self.V = fem.functionspace(self.sim.msh, self.sim.msh.ufl_domain().ufl_coordinate_element())


        self.DG0 = fem.functionspace(self.sim.msh, ("DG", 0))
        self.areaf = ufl.TestFunction(self.DG0)
        self.cell_area_form = fem.form(self.areaf * ufl.dx)
        self.area_0 = np.copy(fem.assemble_vector(self.cell_area_form).array)

        self.area_ratio = fem.Function(self.DG0)
        self.area_ratio.x.array[:] = 1.0


        
        





    def setup(self):
        self.setup_momentum()
        self.setup_solver()

        self.initial_infinite_shelf()
        # self.initial_stokes()


    def initial_infinite_shelf(self):
        # T and P_ice must use the stress scaling of cauchy_stress().
        T = self.sim.params.δ*self.sim.params.ρistar/4

        x = ufl.SpatialCoordinate(self.sim.msh)
        z = x[self.sim.msh.geometry.dim-1]
        P_ice = self.sim.params.ρistar*(1-z)


        sigma_init = ufl.as_tensor([
            [2*T - P_ice, 0,          0],
            [0,          T - P_ice,   0],
            [0,          0,          -P_ice],
        ])

        eps_e_init = (
            0.5 * ufl.dev(sigma_init)
            + ufl.tr(sigma_init)
            / (9 * es.Koverμ(self.sim.params.ν))
            * ufl.Identity(3)
        )

        self.ε_e_prev_time.interpolate(
            fem.Expression(
                eps_e_init,
                self.E.element.interpolation_points(),
            )
        )
        self.ε_e_prev_time.x.scatter_forward()


    def initial_stokes(self):
        u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)
        mixed_el = bufl.mixed_element([u_el,p_el])

        W = fem.functionspace(self.sim.msh, mixed_el)
        w = fem.Function(W)

        du,p = ufl.split(w)

        w_test = ufl.TestFunction(W)
        v,q = ufl.split(w_test)


        εdot = ufl.sym(L(du))/self.sim.params.dtstar
        

        if self.sim.params.n.value==1.0:
            η = 1.0
        else:
            η = mf.viscosity(εdot, self.sim.params.n, 1e-19)
                
        σ = η*εdot - p*ufl.Identity(3)
               
        f = mf.body_force(self.sim.msh)
        n = ufl.FacetNormal(self.sim.msh)

        pw = mf.water_pressure(self.sim.msh, du, self.sim.params.ρwstar, self.sim.params.ucstar, self.sim.params.sea_level_star) + self.sim.params.patmstar
        # pw = mf.water_pressure_static(self.sim.msh, self.sim.params.ρwstar) + self.sim.params.patmstar
        
        
        F = (
            + ufl.inner(σ, ufl.sym(L(v))) - ufl.inner(f, v) 
                ) * ufl.dx 

        if self.sim.basal_friction:
            # water pressure on other boundaries
            F += (
                pw * ufl.inner(n, v) * self.sim.marked_ds(2)\
                )

            # linear weertman on bottom boundary
            F += (
                self.sim.params.βstar*ufl.inner(du/self.sim.params.dtstar,v)*self.sim.marked_ds(1)
            )
        else:
            F += (
                pw * ufl.inner(n, v) * ufl.ds\
                )

        F += (
                        - ufl.div(du)*q \
                        ) * ufl.dx 
        J = ufl.derivative(F,w,ufl.TrialFunction(W))


        problem = solvers.SNESProblem(F, w, bcs=self.sim.bc_funcs[0](W))
        solver = PETSc.SNES().create(MPI.COMM_WORLD)
        
        
        solver.setTolerances(rtol=1.0e-11, max_it=10, atol=1e-13)
        solver.getKSP().setType("preonly")
        # self.solver.getKSP().setTolerances(rtol=1.0e-7)
        solver.getKSP().getPC().setType("lu")
        solver.getKSP().getPC().setFactorSolverType("mumps")
    
        # self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.extract_function_spaces(fem.form(self.F))))
        solver.setFunction(problem.F,dolfinx.fem.petsc.create_vector(fem.form(F)))
        solver.setJacobian(problem.J,dolfinx.fem.petsc.create_matrix(fem.form(J)),P=None)
        solver.solve(None, w.x.petsc_vec)

        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                f"Initial Stokes solve failed: SNES reason {reason}"
            )

        w.x.scatter_forward()

        eps_e_init = (
            0.5 * ufl.dev(σ)
            + ufl.tr(σ) / (9 * es.Koverμ(self.sim.params.ν)) * ufl.Identity(3)
        )

        self.ε_e_prev_time.interpolate(
            fem.Expression(
                eps_e_init,
                self.E.element.interpolation_points(),
            )
        )
        self.ε_e_prev_time.x.scatter_forward()

        
                




    @property
    def pw(self):
        return mf.water_pressure(self.sim.msh, self.du, self.sim.params.ρwstar, self.sim.params.ucstar, self.sim.params.sea_level_star) + self.sim.params.patmstar
        # return mf.water_pressure_static(self.sim.msh, self.sim.params.ρwstar, self.sim.params.sea_level_star) + self.sim.params.patmstar
    
    @property
    def p_crack(self):
        # return mf.water_pressure(self.sim.msh, u, self.sim.params.ρmstar, self.sim.params.ucstar, level=self.sim.params.crack_level_star) + self.sim.params.patmstar
        switch = ufl.conditional(ufl.gt(self.sim.damage.d_prev_it,1e-3),1,0)
        return mf.water_pressure_static(self.sim.msh, self.sim.params.ρwstar,self.sim.params.crack_level_star) + self.sim.params.patmstar
        # return mf.modified_water_pressure(self.sim.msh,
        #             self.sim.params.ρwstar,self.sim.params.ρmstar,
        #             self.sim.params.sea_level_star,
        #             self.sim.params.crack_level_star)



    

    @property
    def σ(self):
        '''Calculate the stress tensor for a given strain tensor and displacement field
        including the effect of water pressure inside cracks.
        The stress is split into a positive and negative part, with the positive part being degraded by the damage variable.'''
        D = ufl.shape(self.ε_e)[0]
        I = ufl.Identity(D); ν = self.sim.params.ν
        
        σplus = es.stress_plus_nt(self.ε_e + self.p_crack*I/(3*es.Koverμ(ν)), ν)
        g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        return g*σplus+ σminus

    @property
    def ψplus(self):
        '''Calculate the positive part of the free energy for a given strain tensor and displacement field,
        including the effect of water pressure inside cracks.'''
        D = ufl.shape(self.ε_e)[0]
        I = ufl.Identity(D); ν = self.sim.params.ν
        return es.free_energy_plus_nt(self.ε_e + self.p_crack*I/(3*es.Koverμ(ν)), ν)
    
    
    def setup_solver(self):


        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)


        self.solver.setTolerances(rtol=1.0e-11, max_it=10, atol=1e-13)
        self.solver.getKSP().setType("preonly")
        # self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")
 
        # self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.extract_function_spaces(fem.form(self.F))))
        self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J,dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)


    def update_bcs(self,new_bcs):
        self.bc_e = new_bcs(self.W)
        # self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)
        self.setup()


    def interpolate_from_parent(self, parent):
        self.area_ratio.interpolate(parent.momentum.area_ratio, cells0=self.sim.parent_cells, cells1=self.sim.cells)


    
    

    def solve(self):
        pass

    def timestep(self):
        pass


    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]
        # self.w_prev_it.x.array[:] = self.w_prev_time.x.array[:]


    def write_checkpoint(self, filename, t=0):
        adios4dolfinx.write_function(filename, self.w, name = "w_momentum",time = t)
        adios4dolfinx.write_function(filename, self.area_ratio, name = "area_ratio", time = t)

    def read_checkpoint(self, filename, t=0):
        adios4dolfinx.read_function(filename, self.w, name = "w_momentum", time = t)
        adios4dolfinx.read_function(filename, self.area_ratio, name = "area_ratio", time = t)


        

