"""Hessian-energy Hageman AT1 models with natural free boundaries.

For the linear mixed squared-Laplacian variants, use higherorder.AT1
or higherorder.AT1Bounded instead.
"""
import numpy as np
import ufl
from dolfinx import fem, default_real_type
from kraken.numerics import solvers
from .higherorder import AugmentedLagrangian, Bounded


class _HagemanAT1Model:
    """Hageman's fourth-order Hessian crack density with Kraken's mechanics.

    Uses HO_Linear2 from Hageman (2026), doi:10.1016/j.cma.2026.118857.
    Quadratic Bernstein C0IP replaces his C1 NURBS discretisation, retaining
    the Hessian energy and its natural boundary/corner terms.
    The dimensional crack density is
        gamma = (2*d + ell**2/2*|grad(d)|**2
                 + ell**4/16*Hess(d):Hess(d)) / (4*ell).

    Damage coefficients (not point values) are bounded; the non-negative
    Bernstein basis then guarantees pointwise bounds and irreversibility.
    The second field is only an L2 projection of the cellwise Laplacian for
    compatibility/diagnostics. It has no feedback into the damage equation.
    Old mixed-CG1 checkpoints cannot be read into this new function space.

    The drive is psi_plus, without threshold subtraction or history.
    Toughness and length are supplied by the calling script, including spatial
    UFL expressions. This model never modifies material parameters.
    """

    def __init__(self, sim):
        self.hessian_penalty = fem.Constant(sim.msh, default_real_type(20.0))
        super().__init__(sim)

    def damage_element(self):
        from .hessian import quadratic_bernstein
        return quadratic_bernstein(self.sim.msh.basix_cell())

    def set_damage_bcs(self, new_bcs):
        # User-specified damage values only. In particular, do not inherit
        # AT2's lap=0 condition or its coordinate-based x=0 exception.
        self.bc_d = list(new_bcs(self.W.sub(0)))

    def hessian_bilinear(self, u, v, coefficient=1):
        from .hessian import hessian_form
        return hessian_form(u, v, self.sim.msh, self.sim.dx, self.hessian_penalty, coefficient)

    def setup_weak_form(self):
        self.H_drive = fem.Function(self.H_space, name="damage driving field")
        self.H_drive_expr = fem.Expression(
            self.driving_energy(), self.H_space.element.interpolation_points(),
        )
        v, q = ufl.TestFunctions(self.W)
        C3, l = self.sim.params.C3, self.sim.params.lstar
        self.F = ((-2*(1-self.d)*v*self.H_drive
                   + self.fracture_residual(v)
                   + (self.lap - ufl.div(ufl.grad(self.d)))*q) * self.sim.dx
                  + self.hessian_bilinear(self.d, v, l**3/(32*C3)))
        self.J = ufl.derivative(self.F, self.w, ufl.TrialFunction(self.W))
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)
        # Bernstein and CG1 basis integrals are strictly positive, unlike
        # the row sums of a quadratic Lagrange mass matrix.
        self._al_weights_form = fem.form((v + q) * self.sim.dx)

    def driving_coefficient(self):
        return 1.0

    def driving_energy(self):
        return self.sim.momentum.ψplus

    def fracture_residual(self, v):
        # Local/gradient contribution; Hessian plus facet terms are assembled
        # separately because they use both cell and interior-facet measures.
        l = self.sim.params.lstar
        weight = 1 / (self.sim.params.C3*l)
        return (weight * 0.5 * v
                + (weight*l**2 / 4) * ufl.inner(ufl.grad(self.d), ufl.grad(v)))

    def fracture_energy(self, d):
        """Discrete fracture energy including supplied toughness and C0IP terms."""
        l = self.sim.params.lstar
        weight = 1 / (self.sim.params.C3*l)
        return (weight*(.5*d + l**2/8*ufl.inner(ufl.grad(d), ufl.grad(d))) * self.sim.dx
                + .5 * self.hessian_bilinear(d, d, weight*l**4/32))



class HagemanAT1(_HagemanAT1Model, AugmentedLagrangian):
    """Hageman fourth-order AT1 with augmented-Lagrangian constraints."""


class HagemanAT1Bounded(_HagemanAT1Model, Bounded):
    """Same Hageman energy and supplied toughness, with PETSc SNES VI.

    Uses reduced-space active-set Newton (vinewtonrsls), enforcing
    d_previous_time <= d <= 1 with an unrestricted auxiliary Laplacian.
    """

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()