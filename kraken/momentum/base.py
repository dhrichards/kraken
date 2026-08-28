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

    @property
    def pw(self):
        return mf.water_pressure(self.sim.msh, self.du, self.sim.params.ρwstar, self.sim.params.ucstar, self.sim.params.sea_level_star) + self.sim.params.patmstar
        # return mf.water_pressure_static(self.sim.msh, self.sim.params.ρwstar, self.sim.params.sea_level_star) + self.sim.params.patmstar
    
    @property
    def p_crack(self):
        # return mf.water_pressure(self.sim.msh, u, self.sim.params.ρmstar, self.sim.params.ucstar, level=self.sim.params.crack_level_star) + self.sim.params.patmstar
        switch = ufl.conditional(ufl.gt(self.sim.damage.d_prev_it,1e-3),1,0)
        return switch*mf.water_pressure_static(self.sim.msh, self.sim.params.ρwstar,self.sim.params.crack_level_star) + self.sim.params.patmstar
        # return mf.modified_water_pressure(self.sim.msh,
        #             self.sim.params.ρwstar,self.sim.params.ρmstar,
        #             self.sim.params.sea_level_star,
        #             self.sim.params.crack_level_star)

    @property
    def u(self):
        return self.u_prev_time + self.du
    
    @property
    def u_v(self):
        return self.u_v_prev_time + self.du_v

    @property
    def u_e(self):
        return self.u_e_prev_time + self.du_e

    @property
    def p(self):
        return self.p_prev_time + self.dp

    @property
    def u_prev_it(self):
        return self.u_prev_time + self.du_prev_it
    
    @property
    def u_v_prev_it(self):
        return self.u_v_prev_time + self.du_v_prev_it

    @property
    def u_e_prev_it(self):
        return self.u_e_prev_time + self.du_e_prev_it

    @property
    def p_prev_it(self):
        return self.p_prev_time + self.dp_prev_it

    @property
    def vel(self):
        return self.du_v / self.sim.params.dtstar

    @property
    def vel_prev_it(self):
        return self.du_v_prev_it / self.sim.params.dtstar

    @property
    def ε_e(self):
        return mf.ε(self.du_e) + self.ε_e_prev_time

    @property
    def ε_e_prev_it(self):
        return mf.ε(self.du_e_prev_it) + self.ε_e_prev_time

    

    @property
    def σ(self):
        '''Calculate the stress tensor for a given strain tensor and displacement field
        including the effect of water pressure inside cracks.
        The stress is split into a positive and negative part, with the positive part being degraded by the damage variable.'''
        I = ufl.Identity(self.sim.msh.geometry.dim); ν = self.sim.params.ν
        
        σplus = es.stress_plus_nt(self.ε_e + self.p_crack*I/(3*es.Koverμ(ν)), ν)
        g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        σminus = σ0 - σplus
        return g*σplus+ σminus

    @property
    def ψplus(self):
        '''Calculate the positive part of the free energy for a given strain tensor and displacement field,
        including the effect of water pressure inside cracks.'''
        I = ufl.Identity(self.sim.msh.geometry.dim); ν = self.sim.params.ν
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
        self.setup_solver()


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


        

