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
    line_search = True
    max_backtracks = 12
    backtracking_armijo = 1e-4
    backtracking_atol = 1e-12

    def setup_momentum(self):
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
        self.creep_newton_weight.value = PETSc.ScalarType(0.0)
        # Start on the prescribed boundary values, so every damped correction
        # has homogeneous boundary data and preserves those values exactly.
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
        self._apply_correction(dq)
        self.w_prev_it.x.array[:] = self.w.x.array
        self.dq_prev_it.x.array[:] = self.dq.x.array

    def _quadrature_rms(self, values):
        # Owned cells only, including participation by ranks with no cells.
        totals = np.array([np.sum(np.abs(values)**2), values.size], dtype=float)
        global_totals = np.empty(2)
        self.sim.msh.comm.Allreduce(totals, global_totals, op=MPI.SUM)
        return np.sqrt(global_totals[0] / max(global_totals[1], 1.0))

    def _step_residual_norms(self):
        # r_local uses the FROZEN fluidity used to construct this Newton step.
        local = (self.r_local_expr.eval(self.sim.msh, self._local_cells)
                 if len(self._local_cells) else np.empty(0))
        return np.array([self._momentum_residual_norm(), self._quadrature_rms(local)])

    def _apply_correction(self, dq):
        u0 = self.w.x.array.copy()
        q0 = self.dq.x.array.copy()
        blocks = self._q_blocks
        self.last_step_length = 1.0
        self.last_backtracks = 0
        if self.line_search:
            initial = self._step_residual_norms()
            if not np.all(np.isfinite(initial)):
                raise RuntimeError("Nonfinite initial mechanics residual")
            # Fixed scales for ALL trials in this search. Including the strain
            # increment avoids dividing by a vanishing initial creep residual.
            scales = np.array([
                max(initial[0], self._vector.norm(), self.backtracking_atol),
                max(initial[1], self._quadrature_rms(q0.reshape(-1, 3)[blocks]),
                    self._quadrature_rms(dq), self.backtracking_atol),
            ])
            self.last_merit_before = float(np.sum((initial / scales)**2))
            roundoff = float(np.sum((self.backtracking_atol / scales)**2))

        for backtrack in range(self.max_backtracks + 1):
            alpha = 0.5**backtrack
            self.w.x.array[:] = u0 + alpha * self.correction.x.array
            self.dq.x.array[:] = q0
            self.dq.x.array.reshape(-1, 3)[blocks] += alpha * dq
            self.dq.x.scatter_forward()
            # u0 and correction have synchronised ghosts; their linear
            # combination needs no additional forward scatter.
            if self.line_search:
                trial = self._step_residual_norms()
                merit = float(np.sum((trial / scales)**2))
                accepted = (np.isfinite(merit) and merit <=
                            (1.0 - self.backtracking_armijo * alpha)
                            * self.last_merit_before + roundoff)
            else:
                accepted = True
            if accepted:
                self.last_step_length = alpha
                self.last_backtracks = backtrack
                if self.line_search:
                    self.last_merit_after = merit
                if backtrack:
                    PETSc.Sys.Print(f"Momentum backtracking: alpha={alpha:.5g}, "
                                    f"merit {self.last_merit_before:.3e} -> {merit:.3e}")
                return

        # Do not expose a rejected mechanical state to the damage solver.
        self.w.x.array[:] = u0
        self.dq.x.array[:] = q0
        self.last_step_length = 0.0
        self.last_backtracks = self.max_backtracks
        raise RuntimeError("Momentum backtracking failed; mechanical state restored")

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