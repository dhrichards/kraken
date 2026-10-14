import adios4dolfinx
from .base import Momentum
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc
from kraken.numerics.invariants import matrix_function
from kraken.boundaryconditions import marked_ds
import basix
from ufl.algorithms import expand_derivatives





class Tensor(Momentum):
    '''
    Class for solving the momentum equation for a Maxwell viscoelastic material

    Time evolution is handled using a semi-Lagrangian approach, such that it solves for the change
    in displacements at each timestep and the mesh is then moved. 

    Solves for the change in total displacement, change in viscous displacement, and pressure as a mixed function.
    The elastic displacement is then calculated as the difference between the total and viscous displacements.
    The elastic strain is then calculated from the elastic displacement and the previous timestep's elastic strain.

    The viscosity can be non-newtonian, for some power law
    '''

    def __init__(self, sim):
        super().__init__(sim)

        msh = self.sim.msh
        cell = msh.basix_cell()
        gdim = msh.geometry.dim

        # ---------------------------------------------------------------
        # Global unknown: displacement increment only
        # ---------------------------------------------------------------
        self.u_el = bufl.element(
            "CG", cell, self.sim.order, shape=(gdim,)
        )

        self.W = fem.functionspace(msh, self.u_el)

        self.w = fem.Function(
            self.W, name="displacement increment"
        )
        self.w_prev_it = fem.Function(
            self.W, name="displacement increment previous iteration"
        )
        self.w_prev_time = fem.Function(
            self.W, name="accumulated displacement"
        )

        # These are aliases, not copies or mixed-space splits.
        self.du = self.w
        self.du_prev_it = self.w_prev_it
        self.u_prev_time = self.w_prev_time

        self.u = self.u_prev_time + self.du

        # The BC function must now accept a vector space, not a mixed space.
        self.bc_e = self.sim.bc_funcs[0](self.W)

        # ---------------------------------------------------------------
        # Quadrature rule and local internal-variable storage
        # ---------------------------------------------------------------
        self.quad_degree = self.sim.quad_degree

        self.q_el = bufl.quadrature_element(
            cell,
            value_shape=(3,),
            scheme="default",
            degree=self.quad_degree,
        )

        self.Q = fem.functionspace(msh, self.q_el)

        # Local viscous increment for the current mechanical iterate.
        # Updated by the local constitutive solver.
        self.dq = fem.Function(
            self.Q, name="local viscous strain increment"
        )

        # Snapshot for your staggered-iteration comparisons.
        self.dq_prev_it = fem.Function(
            self.Q, name="local viscous strain increment previous iteration"
        )

        self.ε_el = bufl.quadrature_element(
            cell,
            value_shape=(3, 3),
            scheme="default",
            degree=self.quad_degree,
        )

        self.E = fem.functionspace(msh, self.ε_el)

        # Accepted physical-timestep history.
        self.ε_e_prev_time = fem.Function(
            self.E, name="previous elastic strain"
        )

        # Temporary storage before committing history.
        self.ε_e_updated = fem.Function(
            self.E, name="updated elastic strain"
        )

        self.dx = self.sim.dx

        # Points used by fem.Expression for local constitutive evaluations.
        self.quad_points, self.quad_weights = basix.make_quadrature(
            cell, self.quad_degree
        )

        # ---------------------------------------------------------------
        # Kinematics: existing scaling and 3D plane-strain embedding
        # ---------------------------------------------------------------
        self.ε_v_dot = (
            mf.viscous_strain(self.dq) / self.sim.params.dtstar
        )

        self.ε_v_dot_prev_it = (
            mf.viscous_strain(self.dq_prev_it) / self.sim.params.dtstar
        )

        self.ε_trial = (
            self.ε_e_prev_time
            + ufl.sym(mf.L(self.du))
        )

        self.ε_trial_prev_it = (
            self.ε_e_prev_time
            + ufl.sym(mf.L(self.du_prev_it))
        )

        self.ε_e = (
            self.ε_trial - mf.viscous_strain(self.dq)
        )

        self.ε_e_prev_it = (
            self.ε_trial_prev_it
            - mf.viscous_strain(self.dq_prev_it)
        )
            

        


        

    def setup_momentum(self):

        v = ufl.TestFunction(self.W)
        dw = ufl.TrialFunction(self.W)

        n = ufl.FacetNormal(self.sim.msh)
        dt = self.sim.params.dtstar

        # Must be a UFL expression depending on self.du and self.dq.
        σ = self.σ

        # Undegraded viscosity.
        if self.sim.params.n.value == 1.0:
            η0 = 1.0
        else:
            # Retains your existing staggered/Picard viscosity treatment.
            η0 = mf.viscosity(
                self.ε_v_dot_prev_it,
                self.sim.params.n,
                1e-19,
            )

        self.η0 = η0

        self.ρ = self.sim.params.ρistar / self.area_ratio
        f = self.ρ * mf.body_force(self.sim.msh)

        # ---------------------------------------------------------------
        # Global mechanical residual
        # ---------------------------------------------------------------
        self.F = (
            ufl.inner(σ, ufl.sym(mf.L(v)))
            - ufl.inner(f, v)
        ) * self.dx

        if self.sim.basal_friction:

            self.F += (
                self.pw * ufl.inner(n, v)
            ) * self.sim.marked_ds(2)

            self.F += (
                self.sim.params.βstar
                * ufl.inner(self.du / dt, v)
            ) * self.sim.marked_ds(1)

        else:

            self.F += (
                self.pw * ufl.inner(n, v)
            ) * ufl.ds

        # ---------------------------------------------------------------
        # Local constitutive residual: three equations per quadrature point
        # ---------------------------------------------------------------
        component_directions = [
            ufl.as_vector((1.0, 0.0, 0.0)),
            ufl.as_vector((0.0, 1.0, 0.0)),
            ufl.as_vector((0.0, 0.0, 1.0)),
        ]

        strain_basis = [
            mf.viscous_strain(ei)
            for ei in component_directions
        ]

        mismatch = (
            η0 * self.ε_v_dot
            - ufl.dev(σ)
        )

        self.r_local = ufl.as_vector([
            ufl.inner(mismatch, Bi)
            for Bi in strain_basis
        ])

        # Local Jacobian A_ij = ∂r_i / ∂dq_j.
        # Use directional derivatives along the three component directions.
        A_columns = [
            expand_derivatives(
                ufl.derivative(
                    self.r_local, self.dq, ei
                )
            )
            for ei in component_directions
        ]

        self.A_local = ufl.as_matrix([
            [A_columns[j][i] for j in range(3)]
            for i in range(3)
        ])

        # Compile expressions for the local Newton solver.
        self.r_local_expr = fem.Expression(
            self.r_local, self.quad_points
        )

        self.A_local_expr = fem.Expression(
            self.A_local, self.quad_points
        )

        # ---------------------------------------------------------------
        # Consistent global Jacobian, including local viscous response
        # ---------------------------------------------------------------

        # Change in local residual caused by a displacement perturbation,
        # keeping dq fixed.
        dr_du = expand_derivatives(
            ufl.derivative(self.r_local, self.w, dw)
        )

        # Implicit differentiation of r_local(dq, u) = 0:
        # δdq = -A^{-1} (∂r/∂u)[δu].
        dq_response = -ufl.dot(
            ufl.inv(self.A_local), dr_du
        )

        # Direct displacement dependence of the global residual.
        J_direct = expand_derivatives(
            ufl.derivative(self.F, self.w, dw)
        )

        # Additional dependence through the locally solved dq.
        J_viscous = expand_derivatives(
            ufl.derivative(
                self.F, self.dq, dq_response
            )
        )

        self.J = J_direct + J_viscous

        self.setup_local_constitutive_solver()

        self.problem = solvers.SNESProblemMOD(
            self.F,
            self.w,
            bcs=self.bc_e,
            J=self.J,
            constitutive_update=self.update_constitutive_state,
        )


    def setup_local_constitutive_solver(self):
        msh = self.sim.msh

        # Solve on owned cells; communicate ghost values afterwards.
        num_owned_cells = msh.topology.index_map(
            msh.topology.dim
        ).size_local

        self._local_cells = np.arange(
            num_owned_cells, dtype=np.int32
        )

        q_dofmap = self.Q.dofmap

        # This implementation uses the blocked shape=(3,) quadrature element.
        if q_dofmap.bs != 3:
            raise ValueError(
                "Expected a quadrature space with block size 3."
            )

        if num_owned_cells:
            self._q_blocks = np.concatenate([
                q_dofmap.cell_dofs(int(cell))
                for cell in self._local_cells
            ])
        else:
            self._q_blocks = np.empty(0, dtype=np.int32)

        expected_points = num_owned_cells * len(self.quad_points)

        if len(self._q_blocks) != expected_points:
            raise ValueError(
                "Quadrature points and local DOF layout do not match."
            )

        # Starting tolerances in your nondimensional residual units.
        self.local_rtol = 1e-10
        self.local_atol = 1e-12
        self.local_max_it = 30

    def update_constitutive_state(self):
        """Solve viscous evolution independently at each quadrature point."""
        msh = self.sim.msh
        cells = self._local_cells
        blocks = self._q_blocks

        # Three stored viscous-strain components per quadrature point.
        q_values = self.dq.x.array.reshape(-1, 3)

        # Deterministic starting guess for each SNES evaluation.
        # This resets the increment only, not the elastic timestep history.
        q_values[blocks] = 0.0

        failed = False

        if len(cells):
            initial_norm = None

            for iteration in range(self.local_max_it + 1):

                r = self.r_local_expr.eval(
                    msh, cells
                ).reshape(-1, 3)

                if not np.all(np.isfinite(r)):
                    failed = True
                    break

                residual_norm = np.linalg.norm(r, axis=1)

                if initial_norm is None:
                    initial_norm = residual_norm.copy()

                tolerance = (
                    self.local_atol
                    + self.local_rtol * initial_norm
                )

                active = residual_norm > tolerance

                if not np.any(active):
                    break

                if iteration == self.local_max_it:
                    failed = True
                    break

                A = self.A_local_expr.eval(
                    msh, cells
                ).reshape(-1, 3, 3)

                if not np.all(np.isfinite(A[active])):
                    failed = True
                    break

                try:
                    correction = np.linalg.solve(
                        A[active],
                        -r[active, :, None],
                    )[..., 0]

                except np.linalg.LinAlgError:
                    failed = True
                    break

                if not np.all(np.isfinite(correction)):
                    failed = True
                    break

                q_values[blocks[active]] += correction

        # All ranks reach this check before any rank raises an exception.
        failed_anywhere = msh.comm.allreduce(failed, op=MPI.LOR)

        if failed_anywhere:
            raise RuntimeError(
                "Local viscous constitutive solve failed to converge."
            )

        self.dq.x.scatter_forward()



    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        # self.problem.solve()
        self.w.x.scatter_forward()
        self.update_constitutive_state()

        self.w_prev_it.x.array[:] = self.w.x.array[:]
        self.w_prev_it.x.scatter_forward()
        self.dq_prev_it.x.array[:] = self.dq.x.array[:]
        self.dq_prev_it.x.scatter_forward()
   

    def timestep(self):

        self.ε_e_updated.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))


        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))

        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))

        self.ε_e_prev_time.x.array[:] = self.ε_e_updated.x.array[:]
        self.w_prev_time.x.array[:] += self.w.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0
        
    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

        
        