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

        self.d_el = bufl.element("CG", self.sim.msh.basix_cell(), self.sim.order)

        # self.H_space = fem.functionspace(self.sim.msh, ("DG", self.sim.order-1))

        self.H_el = bufl.quadrature_element(
            self.sim.msh.basix_cell(),
            value_shape=(),
            scheme="default",
            degree=self.sim.quad_degree,
        )
        self.H_space = fem.functionspace(self.sim.msh, self.H_el)

        
        self.Hprev = fem.Function(self.H_space, name="history")
        self.Hupdated = fem.Function(self.H_space, name="updated history")


    def setup(self):
        self.setup_weak_form()
        self.setup_solver()
        # self.setup_history()

    @property
    def g(self):
        return es.degradation(self.d,self.sim.params.ge_tol)


    def setup_weak_form(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def setup_solver(self):
        '''Set up the nonlinear solver for the damage problem.'''

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.extract_function_spaces(fem.form(self.F))))
        self.solver.setFunction(self.problem.F, dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)
        
        self.solver.setType("newtonls")
        
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")


    def interpolate_from_parent(self, parent):
        '''Interpolate history from parent simulation'''
        self.Hprev.interpolate(parent.damage.Hprev, cells0=self.sim.parent_cells, cells1=self.sim.cells)
        




    
    def timestep(self):
        '''Update the history variable for the damage model.'''
        self.H_func = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
        self.Hupdated.interpolate(
            fem.Expression(
                self.H_func,
                self.sim.momentum.quad_points,
            )
        )
        self.Hupdated.x.scatter_forward()
        self.Hprev.x.array[:] = self.Hupdated.x.array
        self.Hprev.x.scatter_forward()

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
        





