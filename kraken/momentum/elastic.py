import adios4dolfinx
import basix.ufl as bufl
from dolfinx import fem, default_scalar_type
import ufl

from kraken.momentum.base import Momentum
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers


class Elasticity(Momentum):
    """Static elastic adjustment from a fixed initial elastic strain.

    ``u`` is the complete adjustment from that initial state. Neither the
    initial strain nor the mesh is advanced; there is no creep or inertia.
    """

    def __init__(self, sim):
        super().__init__(sim)

        msh = sim.msh
        self.W = fem.functionspace(
            msh, ("Lagrange", sim.order, (msh.geometry.dim,))
        )
        self.u = fem.Function(self.W, name="elastic displacement adjustment")
        self.w = self.du = self.u
        self.u_prev_it = fem.Function(self.W, name="previous displacement iterate")
        self.w_prev_it = self.u_prev_it
        self.u_prev_time = fem.Function(self.W, name="accepted displacement adjustment")
        self.w_prev_time = self.u_prev_time

        ε_el = bufl.quadrature_element(
            msh.basix_cell(), value_shape=(3, 3), scheme=sim.quad_scheme,
            degree=sim.quad_degree,
        )
        self.E = fem.functionspace(msh, ε_el)
        self.quad_points = self.E.element.interpolation_points()
        self.ε_e_initial = fem.Function(self.E, name="initial elastic strain")
        # Alias used by Momentum.initial_infinite_shelf(). This reference
        # strain is deliberately NOT advanced by timestep().
        self.ε_e_prev_time = self.ε_e_initial

        if msh.geometry.dim == 2:
            ε_increment = ufl.sym(mf.L(self.u))
        elif msh.geometry.dim == 3:
            ε_increment = ufl.sym(ufl.grad(self.u))
        else:
            raise ValueError("Elasticity requires a two- or three-dimensional mesh")
        self.ε_e = self.ε_e_initial + ε_increment

        # The shared fixed-point driver sets this; elasticity does not use it.
        self.creep_newton_weight = fem.Constant(msh, default_scalar_type(0.0))

        self.bc_e = sim.bc_funcs[0](self.W)
        self.dx = sim.dx

    @property
    def body_force(self):
        return self.sim.params.ρistar * mf.body_force(self.sim.msh)

    def setup_momentum(self):
        if self.sim.basal_friction:
            raise ValueError(
                "Elasticity is static and does not support velocity-dependent "
                "basal friction; prescribe elastic supports instead."
            )

        v = ufl.TestFunction(self.W)
        ε_test = (
            ufl.sym(mf.L(v)) if self.sim.msh.geometry.dim == 2
            else ufl.sym(ufl.grad(v))
        )
        f = self.body_force
        n = ufl.FacetNormal(self.sim.msh)

        # Use the selected stress and displacement-dependent ocean pressure.
        self.F = (
            ufl.inner(self.σ, ε_test) - ufl.inner(f, v)
        ) * self.dx
        self.F += self.pw * ufl.inner(n, v) * ufl.ds
        self.J = ufl.derivative(self.F, self.u, ufl.TrialFunction(self.W))
        self.problem = solvers.SNESProblem(self.F, self.u, bcs=self.bc_e)

    def solve(self):
        self.solver.solve(None, self.u.x.petsc_vec)
        reason = self.solver.getConvergedReason()
        if reason == -3:
            raise RuntimeError(f"Elastic solve failed: SNES reason {reason}")
        self.u.x.scatter_forward()
        self.u_prev_it.x.array[:] = self.u.x.array
        self.u_prev_it.x.scatter_forward()

    def timestep(self):
        """Accept the adjustment without accumulating strain or moving mesh."""
        self.u_prev_time.x.array[:] = self.u.x.array
        self.u_prev_time.x.scatter_forward()

    def revert(self):
        self.u.x.array[:] = self.u_prev_time.x.array
        self.u.x.scatter_forward()
        self.u_prev_it.x.array[:] = self.u.x.array
        self.u_prev_it.x.scatter_forward()

    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(
            filename, self.ε_e_initial, name="elastic_initial_strain", time=t,
        )
        adios4dolfinx.write_function(
            filename, self.u_prev_time, name="elastic_accepted_adjustment", time=t,
        )

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(
            filename, self.ε_e_initial, name="elastic_initial_strain", time=t,
        )
        adios4dolfinx.read_function(
            filename, self.u_prev_time, name="elastic_accepted_adjustment", time=t,
        )
        self.u.x.scatter_forward()
        self.ε_e_initial.x.scatter_forward()
        self.u_prev_time.x.scatter_forward()
        self.u_prev_it.x.array[:] = self.u.x.array
        self.u_prev_it.x.scatter_forward()


class ElasticityDegradedBodyForce(Elasticity):
    """Elasticity with gravity multiplied by the damage degradation function."""

    @property
    def body_force(self):
        return self.sim.damage.g * super().body_force


class ElasticityHybrid(ElasticityDegradedBodyForce):
    """Degrade both gravity and the whole elastic stress; retain crack pressure.

    The fracture-driving energy is inherited unchanged.
    """

    @property
    def σ(self):
        g = self.sim.damage.g
        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        I = ufl.Identity(self.ε_e.ufl_shape[0])
        return g * σ0 - (1 - g) * self.p_crack * I