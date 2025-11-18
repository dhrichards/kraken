from kraken.numerics import energy_splits as es
from dolfinx import fem
import dolfinx.fem.petsc
from petsc4py import PETSc
from mpi4py import MPI
import ufl
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
        self.setup_history()


    def setup_weak_form(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def setup_solver(self):

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setFunction(self.problem.F, dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)
        
        self.solver.setType("newtonls")
        
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")


    def setup_history(self):
        self.H_func = es.history_function(self.sim.momentum.ε_e, self.Hprev,
                                self.sim.params.ν, self.sim.params.ψcritstar,
                                self.sim.free_energy_plus)
        h = ufl.TrialFunction(self.H_space)
        v = ufl.TestFunction(self.H_space)

        a = ufl.inner(h, v) * ufl.dx
        L = ufl.inner(self.H_func, v) * ufl.dx

        self.history_problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type":"preonly","pc_type":"lu"})




    def timestep(self):
        self.H_func = es.history_function(self.sim.momentum.ε_e, self.Hprev,
                                self.sim.params.ν, self.sim.params.ψcritstar,
                                self.sim.free_energy_plus)
        self.Hprev.interpolate(fem.Expression(self.H_func, self.H_space.element.interpolation_points()))



        # H = self.history_problem.solve()
        # self.Hprev.x.array[:] = H.x.array[:]
        





