from kraken.numerics import energy_splits as es
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
import dolfinx
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import adios4dolfinx

class Damage:
    '''Base class for damage models.'''
    def __init__(self, sim):
        self.sim = sim

        self.d_el = bufl.element("CG", self.sim.msh.basix_cell(), 2)

        self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        self.Hprev = fem.Function(self.H_space, name="history")


    def setup(self):
        self.setup_weak_form()
        self.setup_solver()
        # self.setup_history()


    def setup_weak_form(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def setup_solver(self):
        '''Set up the nonlinear solver for the damage problem.'''

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setFunction(self.problem.F, dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)
        
        self.solver.setType("newtonls")
        
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")
        # self.problem = NonlinearProblem(self.F,self.w,bcs=self.bc_d,
        #                                         petsc_options_prefix='damage_',
        #                                          petsc_options={
        #                                                 "snes_monitor": None,
        #                                                 "ksp_type": "preonly",
        #                                                 "pc_type": "lu",
        #                                                 "pc_factor_mat_solver_type": "mumps",
        #                                                 "ksp_error_if_not_converged": True,
        #                                                 "snes_error_if_not_converged": True,
        #                                          })
        # self.solver = self.problem.solver


    def interpolate_from_parent(self, parent):
        '''Interpolate history from parent simulation'''
        self.Hprev.interpolate(parent.damage.Hprev, cells0=self.sim.parent_cells, cells1=self.sim.cells)
        



    
    def timestep(self):
        '''Update the history variable for the damage model.'''
        self.H_func = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
        self.Hprev.interpolate(fem.Expression(self.H_func, self.H_space.element.interpolation_points()))

    def write_checkpoint(self, filename, t=0):
        adios4dolfinx.write_function(filename, self.w, name = "w_damage",time = t)
        adios4dolfinx.write_function(filename, self.Hprev, name = "Hprev_damage", time = t)
        adios4dolfinx.write_function(filename, self.w_prev_it, name = "w_prev_it_damage", time = t)
        adios4dolfinx.write_function(filename, self.w_prev_it2, name = "w_prev_it2_damage", time = t)

    def read_checkpoint(self, filename, t=0):
        adios4dolfinx.read_function(filename, self.w, name = "w_damage", time = t)
        adios4dolfinx.read_function(filename, self.Hprev, name = "Hprev_damage", time = t)
        adios4dolfinx.read_function(filename, self.w_prev_it, name = "w_prev_it_damage", time = t)
        adios4dolfinx.read_function(filename, self.w_prev_it2, name = "w_prev_it2_damage", time = t)



        # H = self.history_problem.solve()
        # self.Hprev.x.array[:] = H.x.array[:]
        





