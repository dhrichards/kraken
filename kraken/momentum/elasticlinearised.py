"""One elastic Newton correction per outer momentum/damage iteration.

The residual and tangent are inherited from elastic.py, including the stress
split and displacement-dependent ocean pressure. This changes the iteration
strategy, not the elastic equations or the fracture-driving energy.
"""
import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from . import elastic


class Elasticity(elastic.Elasticity):
    """Static elasticity with one MUMPS linear solve per outer iteration."""

    solver_kind = "ksp"

    def setup_momentum(self):
        super().setup_momentum()
        self.correction = fem.Function(self.W, name="elastic correction")
        self._a = fem.form(self.J)
        self._rhs = fem.form(-self.F)
        self._matrix = petsc.create_matrix(self._a)
        self._vector = petsc.create_vector(self._rhs)

    def setup_solver(self):
        self.solver = PETSc.KSP().create(self.sim.msh.comm)
        self.solver.setOptionsPrefix("elasticlinearised_")
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self._matrix)
        self.solver.setFromOptions()

    def solve(self):
        # Linearise at the current displacement and current damage.
        petsc.set_bc(self.u.x.petsc_vec, self.bc_e)
        self.u.x.scatter_forward()
        self._matrix.zeroEntries()
        petsc.assemble_matrix(self._matrix, self._a, bcs=self.bc_e)
        self._matrix.assemble()
        with self._vector.localForm() as local:
            local.set(0.0)
        petsc.assemble_vector(self._vector, self._rhs)
        # Solve J(u,d) delta_u = -F(u,d), with correction boundary values.
        petsc.apply_lifting(self._vector, [self._a], [self.bc_e],
                            x0=[self.u.x.petsc_vec])
        self._vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                 mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self._vector, self.bc_e, x0=self.u.x.petsc_vec)
        self.correction.x.array[:] = 0.0
        self.solver.solve(self._vector, self.correction.x.petsc_vec)
        reason = self.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Linearised elastic solve failed: KSP reason {reason}")
        finite = np.all(np.isfinite(self.correction.x.array))
        if not self.sim.msh.comm.allreduce(bool(finite), op=MPI.LAND):
            raise RuntimeError("Nonfinite linearised elastic correction")
        self.correction.x.scatter_forward()
        # Both arrays have synchronised ghosts, so their sum does too.
        self.u.x.array[:] += self.correction.x.array
        self.u_prev_it.x.array[:] = self.u.x.array


class ElasticityDegradedBodyForce(Elasticity, elastic.ElasticityDegradedBodyForce):
    """Linearised elastic solve with degraded gravity."""


class ElasticityHybrid(Elasticity, elastic.ElasticityHybrid):
    """Linearised elastic solve with degraded gravity and whole stress."""