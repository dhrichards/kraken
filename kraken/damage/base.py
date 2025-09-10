from kraken.numerics import energy_splits as es
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
import basix.ufl as bufl

class Damage:
    def __init__(self, sim):
        self.sim = sim

        self.d_el = bufl.element("CG", self.sim.msh.basix_cell(), 2)

        self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        self.Hprev = fem.Function(self.H_space, name="history")


    def setup(self):
        self.setup_weak_form()
        self.setup_solver()


    def setup_weak_form(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def setup_solver(self):

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(self.J)),P=None)
        
        self.solver.setType("newtonls")
        
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")



    def timestep(self):
        H = es.history_function(self.sim.momentum.ε_e, self.Hprev,
                                self.sim.params.ν, self.sim.params.ψcritstar,
                                self.sim.free_energy_plus)
        self.Hprev.interpolate(fem.Expression(H, self.H_space.element.interpolation_points()))





