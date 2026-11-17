"""One condensed linear mechanics correction per staggered iteration.

Opt in with ``kr.momentum.tensorlinearised.Tensor``. The physical equations,
checkpoint fields and timestep update are inherited from tensorquadrature.
Only n=1 and n=3 are supported, matching its current creep implementation.
"""
import numpy as np
import ufl
from ufl.algorithms import expand_derivatives
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from .tensorquadrature import Tensor as NonlinearTensor
from kraken.numerics import maths_functions as mf


class Tensor(NonlinearTensor):
    solver_kind = "ksp"
    accept_iteration_limit = False
    using_newton = False

    def setup_momentum(self):
        if self.using_newton:
            self.use_linearised()
        # Rebuilding forms/BCs invalidates the cached Newton callbacks.
        for name in ("_newton_solver", "_newton_a", "_newton_b", "_newton_x"):
            if hasattr(self, name):
                getattr(self, name).destroy()
                delattr(self, name)
        if float(self.sim.params.n.value) not in (1.0, 3.0):
            raise ValueError("tensorlinearised supports n=1 or n=3")
        super().setup_momentum()
        self.correction = fem.Function(self.W)

        # Eliminate the local correction from the coupled Newton equations:
        # A δq + B δu = -r, (F_u - F_q A^-1 B) δu = -F + F_q A^-1 r.
        # J already contains the Schur complement, including pressure derivatives.
        local_offset = -ufl.dot(ufl.inv(self.A_local), self.r_local)
        reduced_residual = self.F + expand_derivatives(
            ufl.derivative(self.F, self.dq, local_offset)
        )
        local_correction = -ufl.dot(
            ufl.inv(self.A_local),
            self.r_local + expand_derivatives(
                ufl.derivative(self.r_local, self.w, self.correction)
            ),
        )
        self._correction_expr = fem.Expression(local_correction, self.quad_points)
        self._a = fem.form(self.J)
        self._rhs = fem.form(-reduced_residual)
        self._original_F = fem.form(self.F)
        self._matrix = petsc.create_matrix(self._a)
        self._vector = petsc.create_vector(self._rhs)
        self._residual = petsc.create_vector(self._original_F)

        # Check the unfrozen creep law, not just the linear subproblem.
        s = ufl.dev(self.σ)
        creep = s if float(self.sim.params.n.value) == 1 else ufl.inner(s, s) / 2 * s
        viscous = mf.viscous_strain(self.dq)
        creep_increment = self.sim.params.dtstar * creep
        self._creep_error = fem.Expression(
            ufl.sqrt(ufl.inner(viscous - creep_increment, viscous - creep_increment)),
            self.quad_points,
        )
        self._creep_scale = fem.Expression(
            ufl.sqrt(ufl.inner(viscous, viscous))
            + ufl.sqrt(ufl.inner(creep_increment, creep_increment)),
            self.quad_points,
        )
        self.mechanics_rtol = 1e-8
        self.mechanics_atol = 1e-11

    def setup_solver(self):
        # Called again after timestep rejection; discard the old residual scale.
        self._reference_residual = None
        self.solver = PETSc.KSP().create(self.sim.msh.comm)
        self.solver.setOptionsPrefix("tensorlinearised_")
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self._matrix)
        self.solver.setFromOptions()

    def _momentum_residual_norm(self):
        with self._residual.localForm() as local:
            local.set(0.0)
        petsc.assemble_vector(self._residual, self._original_F)
        self._residual.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                   mode=PETSc.ScatterMode.REVERSE)
        # Strong Dirichlet rows do not participate in the equilibrium residual.
        petsc.set_bc(self._residual, self.bc_e, None, 0.0)
        return self._residual.norm()

    def solve(self):
        if self.using_newton:
            return self._solve_newton()
        self.creep_newton_weight.value = PETSc.ScalarType(0.0)
        # Enforce prescribed values before constructing the correction.
        petsc.set_bc(self.w.x.petsc_vec, self.bc_e)
        self.w.x.scatter_forward()
        # Damage may have changed since the last call.
        self.fluidity_prev_it.interpolate(self._fluidity_expr)
        self.fluidity_prev_it.x.scatter_forward()
        self.fluidity_used.x.array[:] = self.fluidity_prev_it.x.array
        if self._reference_residual is None:
            self._reference_residual = self._momentum_residual_norm()

        self._matrix.zeroEntries()
        petsc.assemble_matrix(self._matrix, self._a, bcs=self.bc_e)
        self._matrix.assemble()
        with self._vector.localForm() as local:
            local.set(0.0)
        petsc.assemble_vector(self._vector, self._rhs)
        # Solve for δu: boundary values are prescribed u minus current u.
        petsc.apply_lifting(self._vector, [self._a], [self.bc_e],
                            x0=[self.w.x.petsc_vec])
        self._vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                 mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self._vector, self.bc_e, x0=self.w.x.petsc_vec)
        self.correction.x.array[:] = 0.0
        self.solver.solve(self._vector, self.correction.x.petsc_vec)
        reason = self.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError("Linearised momentum solve failed: KSP reason "
                               f"{reason}")
        self.correction.x.scatter_forward()

        # Evaluate δq at the OLD state before changing either unknown.
        dq = self._correction_expr.eval(
            self.sim.msh, self._local_cells
        ).reshape(-1, 3) if len(self._local_cells) else np.empty((0, 3))
        finite = np.all(np.isfinite(dq)) and np.all(np.isfinite(self.correction.x.array))
        if not self.sim.msh.comm.allreduce(finite, op=MPI.LAND):
            raise RuntimeError("Nonfinite linearised mechanics correction")
        self.dq.x.array.reshape(-1, 3)[self._q_blocks] += dq
        self.dq.x.scatter_forward()
        self.w.x.array[:] += self.correction.x.array
        self.w_prev_it.x.array[:] = self.w.x.array
        self.dq_prev_it.x.array[:] = self.dq.x.array

    def use_newton(self, rebuild=False):
        """Temporarily use the original local-Newton/global-SNES equations.

        All physical fields stay on this object, so timestep rollback restores
        both solvers consistently. No constitutive-history transfer is needed.
        """
        if rebuild:
            self.use_linearised()
            for name in ("_newton_solver", "_newton_a", "_newton_b", "_newton_x"):
                if hasattr(self, name):
                    getattr(self, name).destroy()
                    delattr(self, name)
        if self.using_newton:
            return
        if not hasattr(self, "_newton_solver"):
            solver = PETSc.SNES().create(self.sim.msh.comm)
            solver.setOptionsPrefix("tensorlinearised_newton_")
            solver.setType("newtonls")
            solver.setTolerances(rtol=1e-11, atol=1e-13, max_it=15)
            solver.getKSP().setType("preonly")
            solver.getKSP().getPC().setType("lu")
            solver.getKSP().getPC().setFactorSolverType("mumps")
            self._newton_b = petsc.create_vector(self.problem.L)
            self._newton_a = petsc.create_matrix(self.problem.a)
            solver.setFunction(self.problem.F, self._newton_b)
            solver.setJacobian(self.problem.J, self._newton_a)
            solver.setFromOptions()
            # Callbacks write into w. Keep the SNES solution separate so line
            # search trial evaluations cannot overwrite its current iterate.
            self._newton_x = self.w.x.petsc_vec.duplicate()
            self._newton_solver = solver
        self._linear_solver = self.solver
        self.solver = self._newton_solver
        self.using_newton = True
        self.solver_kind = "snes"
        self.problem._constitutive_x = None
        # The caller has restored the timestep-start fields before a retry.
        # Recompute the frozen coefficient from those fields, rather than
        # carrying the linearised solver's pre-correction fluidity into Newton.
        self.fluidity_prev_it.interpolate(self._fluidity_expr)
        self.fluidity_prev_it.x.scatter_forward()

    def use_linearised(self):
        if self.using_newton:
            self.solver = self._linear_solver
            self.using_newton = False
            self.solver_kind = "ksp"
        self.problem._constitutive_x = None
        self._reference_residual = None

    def _solve_newton(self):
        # Match tensorquadrature: fluidity remains lagged across outer
        # iterations; the stress split and local viscous equations use Newton.
        self.creep_newton_weight.value = PETSc.ScalarType(0.0)
        self.fluidity_used.x.array[:] = self.fluidity_prev_it.x.array
        self.problem._constitutive_x = None
        self.w.x.petsc_vec.copy(self._newton_x)
        self.solver.solve(None, self._newton_x)
        reason = self.solver.getConvergedReason()
        if reason == -3:
            raise RuntimeError(f"Fallback momentum solve failed: SNES reason {reason}")
        self.problem._update_state(self._newton_x)
        self.w_prev_it.x.array[:] = self.w.x.array
        self.dq_prev_it.x.array[:] = self.dq.x.array
        self.fluidity_prev_it.interpolate(self._fluidity_expr)
        self.fluidity_prev_it.x.scatter_forward()

    def mechanics_converged(self):
        """Called AFTER damage is updated, before accepting an outer iteration."""
        residual = self._momentum_residual_norm()
        tolerance = self.mechanics_atol + self.mechanics_rtol * self._reference_residual
        cells = self._local_cells
        if len(cells):
            error = self._creep_error.eval(self.sim.msh, cells)
            scale = self._creep_scale.eval(self.sim.msh, cells)
            local_ok = bool(np.all(error <= self.mechanics_atol + self.mechanics_rtol * scale))
        else:
            local_ok = True
        return self.sim.msh.comm.allreduce(
            bool(np.isfinite(residual) and residual <= tolerance and local_ok), op=MPI.LAND
        )

    def timestep(self, move_mesh=True):
        super().timestep(move_mesh=move_mesh)
        self._reference_residual = None


class TensorGAMG(Tensor):
    """Optional GMRES/GAMG backend for the same linearised momentum equations.

    Select tensorlinearised.TensorGAMG in model.setup. Tensor retains MUMPS.
    PETSc options use the prefix 'tensorlinearised_gamg_'.
    """

    def setup_solver(self):
        self._reference_residual = None
        self._matrix.setBlockSize(2)
        self._set_rigid_body_modes()
        self.solver = PETSc.KSP().create(self.sim.msh.comm)
        prefix = "tensorlinearised_gamg_"
        self.solver.setOptionsPrefix(prefix)
        self.solver.setType("gmres")
        self.solver.setGMRESRestart(50)
        self.solver.setPCSide(PETSc.PC.Side.RIGHT)
        self.solver.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        self.solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=300)
        self.solver.getPC().setType("gamg")
        self.solver.setOperators(self._matrix)

        self.solver.setFromOptions()

    def _set_rigid_body_modes(self):
        """Two translations and one rotation, globally orthonormalised.

        These are AMG interpolation candidates, not an exact nullspace of the
        constrained, inertial/viscoelastic operator. Include boundary DOFs as
        in the DOLFINx elasticity AMG example.
        """
        if self.sim.msh.geometry.dim != 2 or self.W.dofmap.index_map_bs != 2:
            raise ValueError("TensorGAMG requires the blocked 2D displacement space")
        basis = []
        mode = fem.Function(self.W)
        fields = (lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]))),
                  lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[0]))),
                  lambda x: np.vstack((-x[1], x[0])))
        for field in fields:
            mode.interpolate(field)
            vector = mode.x.petsc_vec.copy()
            # PETSc dot/norm include owned entries only and reduce across MPI.
            for previous in basis:
                vector.axpy(-previous.dot(vector), previous)
            norm = vector.norm()
            if not np.isfinite(norm) or norm == 0:
                raise RuntimeError("Cannot construct displacement rigid-body modes")
            vector.scale(1.0 / norm)
            basis.append(vector)
        near_nullspace = PETSc.NullSpace().create(
            vectors=basis, comm=self.sim.msh.comm
        )
        self._matrix.setNearNullSpace(near_nullspace)
        near_nullspace.destroy()
        for vector in basis:
            vector.destroy()

    def timestep(self, move_mesh=True):
        super().timestep(move_mesh=move_mesh)
        if move_mesh:
            self._set_rigid_body_modes()