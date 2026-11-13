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
from kraken.boundaryconditions import get_zero_bc
from dolfinx.fem.petsc import LinearProblem



class AT2(Damage):
    '''
    Higher order AT2 phase-field damage model.
    This is solved using a mixed formulation, where the damage field and its Laplacian are solved for simultaneously.
    Irreversibility is enforced using a history variable, which is updated at each timestep.
    '''
    def __init__(self, sim):
        super().__init__(sim)
        auxiliary_element = self.d_el
        self.d_el = self.damage_element()
        self.d_el_mixed = bufl.mixed_element([self.d_el, auxiliary_element])

        self.W = fem.functionspace(self.sim.msh, self.d_el_mixed)
        self.w = fem.Function(self.W, name="mixed function")
        self.w_candidate = fem.Function(self.W)
        self.d, self.lap = ufl.split(self.w)
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.d_prev_time, self.lap_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.w_prev_it2 = fem.Function(self.W, name="mixed function 2 iterations previous")
        self.w_prev_it3 = fem.Function(self.W, name="mixed function 3 iterations previous")
        self.d_prev_it, self.lap_prev_it = ufl.split(self.w_prev_it)
        self.d_prev_it2, self.lap_prev_it2 = ufl.split(self.w_prev_it2)
        self.d_prev_it3, self.lap_prev_it3 = ufl.split(self.w_prev_it3)

        self.D, self.d_to_W = self.W.sub(0).collapse()
        self.d_to_W = np.asarray(self.d_to_W, dtype=np.int32)

        # self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        # self.Hprev = fem.Function(self.H_space, name="history")


        self.set_damage_bcs(self.sim.bc_funcs[1])

    def damage_element(self):
        return self.d_el

    def set_damage_bcs(self, new_bcs):
        free_boundary = lambda x: ~np.isclose(x[0], 0.0)
        self.bc_d = list(new_bcs(self.W.sub(0)))
        self.bc_d.append(
            get_zero_bc(self.W.sub(1), free_boundary)
        )

    def update_bcs(self,new_bcs):
        self.set_damage_bcs(new_bcs)
        # self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)
        self.setup()

    def setup_solver(self):
        trial = ufl.TrialFunction(self.W)

        # Replace the current solution with a trial function.
        F_linear = ufl.replace(self.F, {self.w: trial})

        self.problem = LinearProblem(
            ufl.lhs(F_linear),
            ufl.rhs(F_linear),
            u=self.w,
            bcs=self.bc_d,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        self.solver = self.problem.solver


    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        dx = self.sim.dx
        
        
        # H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
        self.H_drive = fem.Function(
            self.H_space, name="damage driving field"
        )

        H_expr = self.driving_energy()

        self.H_drive_expr = fem.Expression(
            H_expr,
            self.H_space.element.interpolation_points(),
        )

        H = self.H_drive

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        # self.F = -C3*2*l*(1-self.d)*v*H*dx \
        #           +(1/2)*(2*self.d*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
        #         ) * dx \
        #         - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * dx
        self.F = (
            -2*C3*l*(1-self.d)*v*H
            + self.fracture_residual(v)
            - self.lap*q
            - ufl.inner(ufl.grad(self.d), ufl.grad(q))
        ) * dx
                

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        # self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def fracture_residual(self, v):
        """Fracture contribution, normalised by Gc / ell."""
        l = self.sim.params.lstar
        return (self.d * v
                + (l**2 / 2) * ufl.inner(ufl.grad(self.d), ufl.grad(v))
                - (l**4 / 16) * ufl.inner(ufl.grad(self.lap), ufl.grad(v)))

    def driving_energy(self):
        return ufl.max_value(
            self.sim.momentum.ψplus - self.sim.params.ψcritstar,
            self.Hprev,
        )

    def interpolate_from_parent(self, parent):
        super().interpolate_from_parent(parent)

        self.w.sub(0).interpolate(parent.damage.w.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w.sub(1).interpolate(parent.damage.w.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        
        self.w_prev_it.sub(0).interpolate(parent.damage.w_prev_it.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it.sub(1).interpolate(parent.damage.w_prev_it.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        self.w_prev_it2.sub(0).interpolate(parent.damage.w_prev_it2.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it2.sub(1).interpolate(parent.damage.w_prev_it2.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        self.w_prev_it3.sub(0).interpolate(parent.damage.w_prev_it3.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it3.sub(1).interpolate(parent.damage.w_prev_it3.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        

        self.w_prev_time.sub(0).interpolate(parent.damage.w_prev_time.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_time.sub(1).interpolate(parent.damage.w_prev_time.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

    def timestep(self):
        super().timestep()
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()



    def solve(self):
        self.w_prev_it3.x.array[:] = self.w_prev_it2.x.array[:]
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]
        # self.solver.solve(None, self.w.x.petsc_vec)
        self.H_drive.interpolate(self.H_drive_expr)
        self.H_drive.x.scatter_forward()

        self.problem.solve()

        # self.w_candidate.x.array[:] = self.w.x.array.copy()
        # self.w.x.array[:] = self.w_prev_it.x.array[:] + 0.5*(self.w_candidate.x.array[:]-self.w_prev_it.x.array[:])
        self.w.x.scatter_forward()
        
        assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"

    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]



class Bounded(AT2):
    '''
    Higher order AT2 phase-field damage model.
    This is solved using a mixed formulation, where the damage field and its Laplacian are solved for simultaneously.
    Uses the signed instantaneous psi_plus - psi_crit driving energy.
    Irreversibility is enforced by d_previous_time <= d <= 1;
    the auxiliary Laplacian has no inequality bounds.
    '''
    def __init__(self, sim):
        super().__init__(sim)
        self.lower_bound = fem.Function(self.W, name="damage_lower_bound")
        self.upper_bound = fem.Function(self.W, name="damage_upper_bound")
        self.previous_bound_tol = 1.0e-3

    

    def driving_energy(self):
        # Signed instantaneous drive: no positive-part clipping or history.
        return self.sim.momentum.ψplus - self.sim.params.ψcritstar

    def setup_weak_form(self):
        # Share AT2's fourth-order mixed form, measure and quadrature field.
        super().setup_weak_form()
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def update_bounds(self):
        # Auxiliary Laplacian remains unrestricted.
        self.lower_bound.x.array[:] = PETSc.NINFINITY
        self.upper_bound.x.array[:] = PETSc.INFINITY

        # Only damage is bounded.
        idx = self.d_to_W
        previous = self.w_prev_time.x.array[idx].copy()
        comm = self.sim.msh.comm
        if comm.allreduce(bool(np.any(~np.isfinite(previous))), op=MPI.LOR):
            raise RuntimeError("Previous damage contains non-finite values; invalid VI bounds")
        violation = comm.allreduce(
            float(np.max(np.maximum(-previous, previous - 1.0), initial=0.0)),
            op=MPI.MAX,
        )
        if violation > self.previous_bound_tol:
            raise RuntimeError(
                f"Previous damage is outside [0, 1]; invalid VI bounds: "
                f"maximum violation {violation:.6e}, "
                f"repair tolerance {self.previous_bound_tol:.6e}"
            )
        self.lower_bound.x.array[idx] = np.clip(previous, 0.0, 1.0)
        self.upper_bound.x.array[idx] = 1.0

        self.lower_bound.x.scatter_forward()
        self.upper_bound.x.scatter_forward()

    def setup_solver(self):
        # AT2.setup_solver creates a linear KSP; bounds require SNES.
        Damage.setup_solver(self)
        # Callbacks copy trial vectors into w; keep the SNES iterate separate.
        self._bounded_x = self.w.x.petsc_vec.duplicate()

        self.update_bounds()

        self.solver.setType("vinewtonrsls")
        self.solver.setVariableBounds(
            self.lower_bound.x.petsc_vec,
            self.upper_bound.x.petsc_vec,
        )

        ksp = self.solver.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        # H_drive is fixed during this solve, so F is affine in (d, lap).
        # Take full reduced-space active-set steps rather than backtracking
        # across changes of the active bounds. PETSc still projects onto the
        # bounds and checks the VI residual; this does not accept divergence.
        # A small initial VI residual can make the relative target fall below
        # assembly roundoff. Set an explicit absolute floor so a residual of
        # order 1e-12 converges instead of exhausting the active-set iterations.
        self.solver.setTolerances(
            rtol=1.0e-9, atol=1.0e-9, stol=0.0, max_it=100,
        )
        prefix = "kraken_bounded_damage_"
        self.solver.setOptionsPrefix(prefix)
        options = PETSc.Options(prefix)
        key = "snes_linesearch_type"
        use_default = not options.hasName(key)
        if use_default:
            options[key] = "basic"
        try:
            # Use options for compatibility with older petsc4py versions
            # that do not expose SNES.getLineSearch().
            self.solver.setFromOptions()
        finally:
            if use_default:
                del options[key]


    def solve(self):
        self.w_prev_it3.x.array[:] = self.w_prev_it2.x.array[:]
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]

        self.update_bounds()

        self.H_drive.interpolate(self.H_drive_expr)
        self.H_drive.x.scatter_forward()

        try:
            self.w.x.petsc_vec.copy(self._bounded_x)
            self.solver.solve(None, self._bounded_x)
            self._bounded_x.copy(self.w.x.petsc_vec)
            self.w.x.scatter_forward()
            reason = self.solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    f"Bounded damage solve failed: SNES reason {reason}, "
                    f"iterations {self.solver.getIterationNumber()}, "
                    f"VI residual {self.solver.getFunctionNorm():.6e}"
                )
        except Exception:
            self.w.x.array[:] = self.w_prev_it.x.array[:]
            self.w.x.scatter_forward()
            raise


    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()



class AugmentedLagrangian(AT2):
    """Higher-order AT2 with signed drive and nodal augmented-Lagrangian bounds.

    Enforces d_previous_time <= d <= 1; the Laplacian remains unrestricted.
    Multipliers are updated outside an ordinary Newton solve. A numerical
    proximal term stabilises negative curvature of the signed-drive problem;
    its anchor is updated each outer iteration, so it vanishes at convergence.
    Neither the drive nor the returned damage field is clipped.
    """

    def __init__(self, sim):
        super().__init__(sim)
        self.al_max_its = 1000
        self.al_rtol = 1.0e-8
        self.al_atol = 1.0e-10
        self.al_bound_tol = 1.0e-8
        # Repair small accumulated feasibility errors in accepted/restarted
        # damage, without relaxing convergence of the next AL solve.
        self.al_previous_bound_tol = 1.0e-3
        self.al_penalty_factor = 1.0
        self.al_iterations = 0

    def driving_energy(self):
        return self.sim.momentum.ψplus - self.sim.params.ψcritstar

    def setup_weak_form(self):
        super().setup_weak_form()
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)
        v, q = ufl.TestFunctions(self.W)
        # Lumped nodal weights scale multiplier forces like the FE residual.
        # This preserves the same nodal constraints as the bounded solver.
        self._al_weights_form = fem.form((v + q) * self.sim.dx)

    def setup_solver(self):
        self.solver = PETSc.SNES().create(self.sim.msh.comm)
        self._al_F = fem.petsc.create_vector(fem.form(self.F))
        self._al_J = fem.petsc.create_matrix(fem.form(self.J))
        self._al_diagonal = self._al_F.duplicate()
        self._al_row_scale = self._al_F.duplicate()
        self._al_original_F = self._al_F.duplicate()
        # Form callbacks copy line-search trial vectors into w. Keep the
        # Newton iterate separate, otherwise a trial overwrites its base point.
        self._al_x = self.w.x.petsc_vec.duplicate()
        self.solver.setFunction(self._al_residual, self._al_F)
        self.solver.setJacobian(self._al_jacobian, self._al_J)
        self.solver.setType("newtonls")
        # Each multiplier update must get a primal correction. Otherwise a
        # tiny scaled initial residual can skip Newton while bounds still
        # fail, causing needless penalty growth in high-contrast fields.
        self.solver.setForceIteration(True)
        # The frozen, proximal-stabilised subproblem is piecewise affine.
        # Residual backtracking can reject a step when a bound activates.
        # Use semismooth Newton full steps, retaining residual convergence
        # checks and the independent final KKT checks below.
        prefix = "kraken_al_damage_"
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
        # The inner solve need only be more accurate than the final KKT
        # tolerance. An absolute 1e-10 was below the assembly/solve noise
        # floor on shelf meshes, despite the final target being about 1e-8.
        self.solver.setTolerances(
            rtol=1.0e-10, atol=0.1 * (self.al_atol + self.al_rtol),
            stol=0.0, max_it=80,
        )
        ksp = self.solver.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

    def _constraint_terms(self):
        d = self.w.x.array[self._al_idx]
        lower = self._al_λ_lower + self._al_penalty * (self._al_lower - d)
        upper = self._al_λ_upper + self._al_penalty * (d - 1.0)
        return lower, upper

    def _al_residual(self, snes, x, F):
        self.problem.F(snes, x, F)
        lower, upper = self._constraint_terms()
        d = self.w.x.array[self._al_idx]
        force = (-np.maximum(lower, 0.0) + np.maximum(upper, 0.0)
                 + self._al_proximal * (d - self._al_anchor))
        F.array[self._al_idx] += self._al_weights[self._al_idx] * force
        F.pointwiseMult(F, self._al_row_scale)

    def _al_jacobian(self, snes, x, J, P):
        # Do not depend on the residual callback having updated self.w first.
        x.copy(self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        self.problem.J(snes, x, J, P)
        lower, upper = self._constraint_terms()
        active = (lower > 0.0).astype(float) + (upper > 0.0).astype(float)
        self._al_diagonal.set(0.0)
        self._al_diagonal.array[self._al_idx] = (
            self._al_weights[self._al_idx]
            * (self._al_penalty * active + self._al_proximal)
        )
        J.setDiagonal(self._al_diagonal, addv=PETSc.InsertMode.ADD_VALUES)
        J.assemble()
        J.diagonalScale(self._al_row_scale, None)

    def _global_max(self, values):
        return self.sim.msh.comm.allreduce(
            float(np.max(values, initial=0.0)), op=MPI.MAX
        )

    def _prepare_augmented_solve(self):
        # Reassemble after mesh movement. Work only on owned dofs; PETSc and
        # the original form callbacks handle all ghost communication.
        weights = fem.petsc.assemble_vector(self._al_weights_form)
        weights.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        self._al_weights = weights.array_r.copy()
        weights.destroy()
        owned = len(self._al_weights)
        fixed = np.zeros(owned, dtype=bool)
        for bc in self.bc_d:
            dofs, _ = bc.dof_indices()
            fixed[dofs[dofs < owned]] = True
        self._al_free = np.flatnonzero(~fixed)
        idx = self.d_to_W[self.d_to_W < owned]
        self._al_idx = idx[~fixed[idx]]
        self._al_lower = self.w_prev_time.x.array[self._al_idx].copy()
        if self._global_max(~np.isfinite(self._al_lower)):
            raise RuntimeError("Previous damage contains non-finite values; invalid AL bounds")
        violation = self._global_max(
            np.maximum(-self._al_lower, self._al_lower - 1.0)
        )
        if violation > self.al_previous_bound_tol:
            raise RuntimeError(
                "Previous damage is outside [0, 1]; invalid AL bounds: "
                f"maximum violation {violation:.6e}, "
                f"repair tolerance {self.al_previous_bound_tol:.6e}"
            )
        # Even an accepted d = 1 + epsilon would otherwise give incompatible
        # lower/upper bounds. Negative errors can also accumulate over steps.
        # Only repair the previous damage; leave the auxiliary field and the
        # signed driving energy untouched.
        np.clip(self._al_lower, 0.0, 1.0, out=self._al_lower)
        self.w_prev_time.x.array[self._al_idx] = self._al_lower
        self.w_prev_time.x.scatter_forward()

        C3 = float(self.sim.params.C3)
        l = float(self.sim.params.lstar)
        drive = 2.0 * C3 * l * self.H_drive.x.array
        self._al_scale = max(1.0, self._global_max(np.abs(drive)))
        # Use the same nodal scaling as the final stationarity check. Apply
        # it to both residual and Jacobian; the Newton equation is unchanged.
        # Dirichlet rows keep their unit scaling.
        self._al_row_scale.set(1.0)
        self._al_row_scale.array[self._al_free] = 1.0 / self._al_weights[self._al_free]
        self._al_row_scale.array[self._al_idx] /= self._al_scale
        # For CG1, lumped mass dominates the consistent mass in quadratic
        # form. This shift offsets negative local curvature without changing
        # the final AT2 equation, unlike adding physical damage viscosity.
        self._al_proximal = self._global_max(np.maximum(-drive, 0.0))
        # Large positive drive already provides positive curvature. Using its
        # global maximum for the penalty over-penalises weakly driven nodes
        # and can make the inner active set cycle.
        self._al_penalty = self.al_penalty_factor * max(1.0, self._al_proximal)
        if self._al_penalty <= 0.0:
            raise ValueError("al_penalty_factor must be positive")
        self._al_anchor = self.w.x.array[self._al_idx].copy()
        self.problem.F(self.solver, self.w.x.petsc_vec, self._al_original_F)
        reaction = (self._al_original_F.array_r[self._al_idx]
                    / self._al_weights[self._al_idx])
        d = self._al_anchor
        self._al_λ_lower = np.where(d <= self._al_lower + self.al_bound_tol,
                                   np.maximum(reaction, 0.0), 0.0)
        self._al_λ_upper = np.where(d >= 1.0 - self.al_bound_tol,
                                   np.maximum(-reaction, 0.0), 0.0)

    def _check_augmented_convergence(self):
        d = self.w.x.array[self._al_idx]
        self.al_constraint_violation = self._global_max(
            np.maximum(self._al_lower - d, d - 1.0)
        )
        self.problem.F(self.solver, self.w.x.petsc_vec, self._al_original_F)
        residual = self._al_original_F.array_r.copy() / self._al_weights
        residual[self._al_idx] -= self._al_λ_lower - self._al_λ_upper
        residual[self._al_idx] /= self._al_scale
        self.al_residual = self._global_max(np.abs(residual[self._al_free]))
        complementarity = np.maximum(
            np.abs(self._al_λ_lower * (d - self._al_lower)),
            np.abs(self._al_λ_upper * (1.0 - d)),
        ) / self._al_scale
        self.al_complementarity = self._global_max(complementarity)
        tol = self.al_atol + self.al_rtol
        return (self.al_constraint_violation <= self.al_bound_tol
                and self.al_residual <= tol
                and self.al_complementarity <= tol)

    def solve(self):
        self.w_prev_it3.x.array[:] = self.w_prev_it2.x.array[:]
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]
        self.H_drive.interpolate(self.H_drive_expr)
        self.H_drive.x.scatter_forward()
        self._prepare_augmented_solve()
        # Respect changes to the outer tolerance made after setup as well.
        self.solver.setTolerances(atol=0.1 * (self.al_atol + self.al_rtol))
        previous_violation = np.inf
        self.al_penalty_updates = 0
        try:
            for iteration in range(1, self.al_max_its + 1):
                self.al_iterations = iteration
                self._al_anchor[:] = self.w.x.array[self._al_idx]
                self.w.x.petsc_vec.copy(self._al_x)
                self.solver.solve(None, self._al_x)
                self._al_x.copy(self.w.x.petsc_vec)
                self.w.x.scatter_forward()
                reason = self.solver.getConvergedReason()
                if reason <= 0:
                    raise RuntimeError(
                        f"AL inner solve failed at outer iteration {iteration}: "
                        f"SNES reason {reason}, iterations {self.solver.getIterationNumber()}, "
                        f"scaled residual {self.solver.getFunctionNorm():.6e}, "
                        f"penalty {self._al_penalty:.6e}, "
                        f"proximal shift {self._al_proximal:.6e}"
                    )
                lower, upper = self._constraint_terms()
                self._al_λ_lower[:] = np.maximum(lower, 0.0)
                self._al_λ_upper[:] = np.maximum(upper, 0.0)
                if self._check_augmented_convergence():
                    return
                # Increase the penalty only when feasibility stalls. Starting
                # with the global positive-drive maximum made the first inner
                # solve needlessly ill-conditioned in heterogeneous fields.
                if iteration % 20 == 0:
                    if (self.al_constraint_violation > self.al_bound_tol
                            and self.al_constraint_violation > 0.5 * previous_violation):
                        self._al_penalty *= 5.0
                        self.al_penalty_updates += 1
                    previous_violation = self.al_constraint_violation
            raise RuntimeError(
                f"AL damage solve failed after {self.al_max_its} iterations: "
                f"stationarity {self.al_residual:.6e}, "
                f"bound violation {self.al_constraint_violation:.6e}, "
                f"complementarity {self.al_complementarity:.6e}"
            )
        except Exception:
            self.w.x.array[:] = self.w_prev_it.x.array[:]
            self.w.x.scatter_forward()
            raise

    def timestep(self):
        # Irreversibility is tied to the accepted timestep, not the outer
        # multiplier iterations or the momentum/damage fixed-point loop.
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()


class _AT1Model:
    """Mixed CG1 higher-order AT1 with squared-Laplacian regularisation.

    Retains the original mixed formulation and its lap=0 free-boundary
    condition. Uses instantaneous psi_plus and the strength calibration
    Gc=2*ell*sigma_t**2/E (overwriting Kic), as before the Hessian change.
    This is distinct from Hageman's Hessian model in damage.hageman.
    """

    def __init__(self, sim):
        super().__init__(sim)
        self.update_toughness()

    def update_toughness(self):
        p = self.sim.params
        ell, strength, E, nu = map(float, (p.l, p.σt, p.E, p.ν))
        if (not np.all(np.isfinite([ell, strength, E, nu]))
                or ell <= 0.0 or strength <= 0.0 or E <= 0.0
                or not -1.0 < nu < 0.5):
            raise ValueError("AT1 requires positive ell, sigma_t, E "
                             "and -1 < nu < 0.5")
        # Gc = Kic**2*(1-nu**2)/E = 2*ell*sigma_t**2/E.
        # Keep the Constant itself: previously built UFL forms refer to it.
        p.Kic.value = np.sqrt(2.0 * ell * strength**2 / (1.0 - nu**2))

    def setup_weak_form(self):
        self.update_toughness()
        super().setup_weak_form()

    def driving_energy(self):
        return self.sim.momentum.ψplus

    def fracture_residual(self, v):
        l = self.sim.params.lstar
        return (0.5 * v
                + (l**2 / 4) * ufl.inner(ufl.grad(self.d), ufl.grad(v))
                - (l**4 / 32) * ufl.inner(ufl.grad(self.lap), ufl.grad(v)))

    def solve(self):
        self.update_toughness()
        super().solve()


class AT1(_AT1Model, AugmentedLagrangian):
    """Mixed squared-Laplacian AT1 with augmented-Lagrangian constraints."""


class AT1Bounded(_AT1Model, Bounded):
    """Mixed squared-Laplacian AT1 with PETSc reduced-space VI constraints."""

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()