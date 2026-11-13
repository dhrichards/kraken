import adios4dolfinx
from .base import Damage
import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type, nls
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics.maths_functions import ε
from kraken.numerics import solvers
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np



class LowerOrder(Damage):
    ''' Parent class for lower order damage models, such as AT1 and AT2'''
    def __init__(self, sim):
        super().__init__(sim)


        self.D = fem.functionspace(self.sim.msh, self.d_el)
        self.d = fem.Function(self.D, name="damage")
        self.w = self.d # for compatibility
        self.d_prev_time = fem.Function(self.D, name="damage_prev_time")
        self.w_prev_time = self.d_prev_time  # for compatibility

        self.d_prev_it = fem.Function(self.D, name="damage_prev_it")
        self.d_prev_it2 = fem.Function(self.D, name="damage_prev_it2")
        self.d_prev_it3 = fem.Function(self.D, name="damage_prev_it3")

        self.w_prev_it = self.d_prev_it
        self.w_prev_it2 = self.d_prev_it2
        self.w_prev_it3 = self.d_prev_it3


        

        self.bc_d = self.sim.bc_funcs[1](self.D)
        

    def solve(self):
        self.d_prev_it3.x.array[:] = self.d_prev_it2.x.array[:]
        self.d_prev_it2.x.array[:] = self.d_prev_it.x.array[:]
        self.d_prev_it.x.array[:] = self.d.x.array[:]
        self.solver.solve(None, self.d.x.petsc_vec)

        self.d.x.scatter_forward()
        
        assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"


   



class AT2(LowerOrder):
    '''Lower oder AT2 phase-field damage model.
      Irreversibility is enforced using a history variable, which is updated at each timestep.'''
    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
    

        v = ufl.TestFunction(self.D)

        
        self.F = (ufl.inner(self.d,v) + l**2*ufl.inner(ufl.grad(self.d), ufl.grad(v)) \
                - C3*l*2*(1-self.d)*H*v) * ufl.dx
        

        self.J = ufl.derivative(self.F,self.d,ufl.TrialFunction(self.D))


        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)


class AT2Bounded(AT2):


    def setup_weak_form(self):
        C3 = self.sim.params.C3
        l = self.sim.params.lstar
        dx = self.sim.momentum.dx

        H = ufl.max_value(
            self.sim.momentum.ψplus - self.sim.params.ψcritstar, 0.0
        )

        v = ufl.TestFunction(self.D)

        self.F = (
            self.d*v
            + l**2*ufl.inner(ufl.grad(self.d), ufl.grad(v))
            - 2*C3*l*(1-self.d)*H*v
        ) * dx

        self.J = ufl.derivative(
            self.F, self.d, ufl.TrialFunction(self.D)
        )
        self.problem = solvers.SNESProblem(
            self.F, self.d, bcs=self.bc_d
        )


    def __init__(self, sim):
        super().__init__(sim)
        self.previous_bound_tol = 1.e-3
        self.lower_bound = fem.Function(self.D, name="damage_lower_bound")
        self.upper_bound = fem.Function(self.D, name="damage_upper_bound")
        self.upper_bound.x.array[:] = 1.

    def update_bounds(self):
        previous = self.d_prev_time.x.array
        comm = self.sim.msh.comm
        if comm.allreduce(bool(np.any(~np.isfinite(previous))), op=MPI.LOR):
            raise RuntimeError("Previous damage contains non-finite values; invalid VI bounds")
        violation = comm.allreduce(
            float(np.max(np.maximum(-previous, previous-1.), initial=0.)), op=MPI.MAX)
        if violation > self.previous_bound_tol:
            raise RuntimeError(
                f"Previous damage is outside [0, 1]; invalid VI bounds: "
                f"maximum violation {violation:.6e}, "
                f"repair tolerance {self.previous_bound_tol:.6e}")
        self.lower_bound.x.array[:] = np.clip(previous, 0., 1.)
        self.lower_bound.x.scatter_forward()
        self.upper_bound.x.scatter_forward()

    def setup_solver(self):
        self.solver = PETSc.SNES().create(self.sim.msh.comm)
        self._F = fem.petsc.create_vector(fem.form(self.F))
        self._J = fem.petsc.create_matrix(fem.form(self.J))
        self._bounded_x = self.d.x.petsc_vec.duplicate()
        self.solver.setFunction(self.problem.F, self._F)
        self.solver.setJacobian(self.problem.J, self._J, P=None)
        self.update_bounds()
        self.solver.setType("vinewtonrsls")
        self.solver.setVariableBounds(self.lower_bound.x.petsc_vec,
                                      self.upper_bound.x.petsc_vec)
        self.solver.setTolerances(rtol=1.e-9, atol=1.e-12, stol=0., max_it=100)
        ksp = self.solver.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        prefix = "kraken_lowerorder_bounded_damage_"
        self.solver.setOptionsPrefix(prefix)
        options = PETSc.Options(prefix)
        key = "snes_linesearch_type"
        use_default = not options.hasName(key)
        if use_default:
            options[key] = "basic"
        try:
            self.solver.setFromOptions()
        finally:
            if use_default:
                del options[key]

    def solve(self):
        self.d_prev_it3.x.array[:] = self.d_prev_it2.x.array[:]
        self.d_prev_it2.x.array[:] = self.d_prev_it.x.array[:]
        self.d_prev_it.x.array[:] = self.d.x.array[:]
        self.update_bounds()
        try:
            self.d.x.petsc_vec.copy(self._bounded_x)
            self.solver.solve(None, self._bounded_x)
            self._bounded_x.copy(self.d.x.petsc_vec)
            self.d.x.scatter_forward()
            reason = self.solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    f"Lower-order bounded damage solve failed: SNES reason {reason}, "
                    f"iterations {self.solver.getIterationNumber()}, "
                    f"VI residual {self.solver.getFunctionNorm():.6e}")
        except Exception:
            self.d.x.array[:] = self.d_prev_it.x.array[:]
            self.d.x.scatter_forward()
            raise

    def timestep(self):
        self.d_prev_time.x.array[:] = self.d.x.array[:]
        self.d_prev_time.x.scatter_forward()

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        self.d_prev_time.x.array[:] = self.d.x.array[:]
        self.d_prev_time.x.scatter_forward()

    def update_bcs(self, new_bcs):
        self.bc_d = list(new_bcs(self.D))
        self.setup()


class AT1(AT2Bounded):
    """Standard second-order AT1 with bounded, irreversible CG1 damage.

    gamma = 3/8 * (d/ell + ell*|grad(d)|**2).
    Uses instantaneous psi_plus, with no threshold subtraction or history.
    The strength calibration overwrites the existing Kic Constant in place:
      Gc = (8/3)*(1-ge_tol)*ell*sigma_t**2/E.
    This makes homogeneous onset occur at psi_plus=sigma_t**2/(2E).
    Free boundaries have zero normal damage gradient; prescribed damage
    values continue to come from the user's boundary callback.
    """

    def __init__(self, sim):
        super().__init__(sim)
        self.update_toughness()

    def update_toughness(self):
        p = self.sim.params
        ell, strength, E, nu, k = map(float, (p.l, p.σt, p.E, p.ν, p.ge_tol))
        if (not np.all(np.isfinite([ell, strength, E, nu, k]))
                or ell <= 0. or strength <= 0. or E <= 0.
                or not -1. < nu < .5 or not 0. <= k < 1.):
            raise ValueError("AT1 requires positive ell, sigma_t, E, "
                             "-1 < nu < 0.5 and 0 <= ge_tol < 1")
        # Retain the Constant referenced by existing mechanics/UFL forms.
        p.Kic.value = np.sqrt((8./3.)*(1.-k)*ell*strength**2/(1.-nu**2))

    def setup_weak_form(self):
        self.update_toughness()
        C3, l = self.sim.params.C3, self.sim.params.lstar
        dx = self.sim.dx
        self.total_energy = (
            (1/C3)*es.crack_density_function(self.d, l, lambda d: d, 8./3.)
            + self.g*self.sim.momentum.ψplus
        ) * dx
        # Positive, spatially constant normalisation, matching the scale of
        # the AT2 residual. Does not change stationarity or the VI solution.
        self.F = C3*l*ufl.derivative(
            self.total_energy, self.d, ufl.TestFunction(self.D))
        self.J = ufl.derivative(self.F, self.d, ufl.TrialFunction(self.D))
        self.problem = solvers.SNESProblem(self.F, self.d, bcs=self.bc_d)

    def solve(self):
        self.update_toughness()
        super().solve()