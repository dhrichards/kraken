"""Experimental JAX adapter for the current 2D tensorquadrature backend.

Explicit opt-in: from kraken.momentum.tensorjax import TensorJAX.
Full 3D material support lives in jax_material; the mesh adapter remains 2D.
"""
import numpy as np
import basix.ufl as bufl
import ufl
from dolfinx import fem
from mpi4py import MPI
from .tensorquadrature import Tensor
from kraken.numerics import maths_functions as mf, solvers
from kraken.numerics import jax_material as material


def mandel(A):
    r = np.sqrt(2.)
    return ufl.as_vector((A[0,0], A[1,1], A[2,2], r*A[1,2], r*A[0,2], r*A[0,1]))


def tensor(a):
    r = np.sqrt(2.)
    return ufl.as_tensor(((a[0],a[5]/r,a[4]/r),
                          (a[5]/r,a[1],a[3]/r),
                          (a[4]/r,a[3]/r,a[2])))


class TensorJAX(Tensor):
    def __init__(self, sim):
        if sim.msh.geometry.dim != 2:
            raise NotImplementedError('The JAX material supports 3D; this first mesh adapter is 2D only.')
        super().__init__(sim)
        self.σ_q = self._quadrature_field((6,), 'JAX stress')
        self.C_alg_q = self._quadrature_field((4,4), 'JAX algorithmic tangent')
        self.ψplus_q = self._quadrature_field((), 'JAX positive energy')
        self.σplus_q = self._quadrature_field((6,), 'JAX positive stress')
        self.g_used_q = self._quadrature_field((), 'JAX degradation snapshot')
        self._integrate = material.make_integrator(
            dimension=2, reduced_tangent=True,
            batch_size=getattr(sim, 'jax_batch_size', None))
        self._field_blocks = {}

    def _quadrature_field(self, shape, name):
        element = bufl.quadrature_element(self.sim.msh.basix_cell(),
            value_shape=shape, scheme=self.sim.quad_scheme, degree=self.quad_degree)
        return fem.Function(fem.functionspace(self.sim.msh, element), name=name)

    @property
    def σ(self):
        """Cached σ. Call refresh_material_outputs after manual field edits."""
        return tensor(self.σ_q) + (self.sim.damage.g-self.g_used_q)*tensor(self.σplus_q)

    @property
    def ψplus(self):
        return self.ψplus_q

    def setup_momentum(self):
        self.setup_local_constitutive_solver()  # inherited tolerances and owned-cell layout
        params = self.sim.params
        ε_trial_mandel = mandel(self.ε_trial)
        self._inputs = fem.Expression(ufl.as_vector(
            [ε_trial_mandel[i] for i in range(6)] + [self.sim.damage.g, self.p_crack,
             params.ν, params.dtstar, params.n, self.creep_newton_weight, self.fluidity_prev_it]), self.quad_points)
        ε_e_mandel = mandel(self.ε_e)
        self._diagnostic_inputs = fem.Expression(ufl.as_vector(
            [ε_e_mandel[i] for i in range(6)] + [self.sim.damage.g, self.p_crack, params.ν]), self.quad_points)
        v = ufl.TestFunction(self.W)
        dw = ufl.TrialFunction(self.W)
        n = ufl.FacetNormal(self.sim.msh)
        self.ρ = params.ρistar/self.area_ratio
        self.F = (ufl.inner(self.σ, ufl.sym(mf.L(v)))
                  - ufl.inner(self.ρ*mf.body_force(self.sim.msh), v))*self.dx
        if self.sim.basal_friction:
            self.F += self.pw*ufl.inner(n,v)*self.sim.marked_ds(2)
            self.F += params.βstar*ufl.inner(self.du/params.dtstar,v)*self.sim.marked_ds(1)
        else:
            self.F += self.pw*ufl.inner(n,v)*ufl.ds
        self.F += params.C_inertia*self.ρ*ufl.inner(self.acceleration,v)*self.dx
        self.F += self.momentum_damping*ufl.inner(self.w-self.w_event_prev,v)*self.dx
        # UFL differentiates external loads, inertia and damping. Stress is a
        # coefficient, so its missing derivative is supplied explicitly below.
        direct = ufl.derivative(self.F, self.w, dw)
        ε_trial_plane = ufl.as_vector([ε_trial_mandel[i] for i in (0,1,2,4)])
        test_mandel = mandel(ufl.sym(mf.L(v)))
        test_plane = ufl.as_vector([test_mandel[i] for i in (0,1,2,4)])
        trial_variation = ufl.derivative(ε_trial_plane, self.w, dw)
        self.J = direct + ufl.inner(test_plane,
                    ufl.dot(self.C_alg_q, trial_variation))*self.dx
        self.problem = solvers.SNESProblemMOD(self.F, self.w, self.bc_e,
                         J=self.J, constitutive_update=self.update_constitutive_state)
        self.refresh_material_outputs()

    def _blocks(self, field):
        key = id(field.function_space)
        if key not in self._field_blocks:
            dm = field.function_space.dofmap
            blocks = (np.concatenate([dm.cell_dofs(int(c)) for c in self._local_cells])
                      if len(self._local_cells) else np.empty(0, dtype=np.int32))
            if len(blocks) != len(self._local_cells)*len(self.quad_points):
                raise ValueError('Unexpected quadrature field layout')
            self._field_blocks[key] = blocks
        return self._field_blocks[key]

    def _write(self, field, values):
        width = int(np.prod(field.ufl_shape)) if field.ufl_shape else 1
        field.x.array.reshape(-1,width)[self._blocks(field)] = np.asarray(values).reshape(-1,width)
        field.x.scatter_forward()

    def _collective_failure(self, failure):
        comm = self.sim.msh.comm
        if comm.allreduce(failure is not None, op=MPI.LOR):
            reports = comm.allgather(None if failure is None else f'rank {comm.rank}: {failure}')
            raise RuntimeError('JAX material evaluation failed: '+ '; '.join(x for x in reports if x))

    def update_constitutive_state(self):
        """Stage results, then commit only if all MPI ranks succeeded."""
        count = len(self._q_blocks)
        result = None
        g = np.empty(0)
        failure = None
        try:
            if count:
                data = np.asarray(self._inputs.eval(self.sim.msh, self._local_cells)).reshape(count,13)
                if not np.all(np.isfinite(data)):
                    raise ValueError('nonfinite material inputs')
                if not np.all(np.isin(data[:,10], [1.,3.])):
                    raise ValueError('Only the current n=1 and n=3 creep laws are supported')
                if np.any(data[:,9] <= 0) or np.any((data[:,8] <= -1) | (data[:,8] >= .5)):
                    raise ValueError('Require dtstar > 0 and -1 < ν < 0.5')
                if np.any((data[:,11] < 0) | (data[:,11] > 1)) or np.any(data[:,12] < 0):
                    raise ValueError('Require θ in [0,1] and nonnegative fluidity')
                g = data[:,6]
                dq_guess = self.dq_prev_it.x.array.reshape(-1,3)[self._q_blocks]
                result = tuple(np.asarray(x) for x in self._integrate(
                    data[:,:6], data[:,6:], dq_guess, self.local_atol, self.local_rtol))
                if not np.all(result[5]):
                    raise RuntimeError(f'{np.count_nonzero(~result[5])} failed points; '
                                       f'max residual/tolerance={np.max(result[4]):.3e}')
        except Exception as exc:
            failure = str(exc)
        self._collective_failure(failure)
        if result is None:
            result = (np.empty((0,3)),np.empty((0,6)),np.empty(0),np.empty((0,4,4)),None,None,None,np.empty((0,6)))
        self._write(self.dq, result[0])
        self._write(self.σ_q, result[1])
        self._write(self.ψplus_q, result[2])
        self._write(self.C_alg_q, result[3])
        self._write(self.σplus_q, result[7])
        self._write(self.g_used_q, g)

    def refresh_material_outputs(self):
        """Refresh stress/energy at fixed dq, without solving or advancing history.

        Use after initialisation, damage changes or loading plotting snapshots.
        Invalidates the mechanical cache: this does not refresh the condensed tangent.
        """
        failure = None
        ψplus, σ = np.empty(0), np.empty((0,6))
        σplus, g = np.empty((0,6)), np.empty(0)
        try:
            count = len(self._q_blocks)
            if count:
                a = np.asarray(self._diagnostic_inputs.eval(self.sim.msh, self._local_cells)).reshape(count,9)
                if np.any(a[:,[3,5]] != 0):
                    raise ValueError('Plane-strain material requires ε_yz = ε_xy = 0')
                g = a[:,6]
                ψplus, σ, σplus = (np.asarray(x) for x in material.evaluate_batch_plane_strain(a[:,:6],a[:,6],a[:,7],a[:,8]))
                if not all(np.all(np.isfinite(x)) for x in (ψplus,σ,σplus,g)):
                    raise ValueError('nonfinite diagnostic fields')
        except Exception as exc:
            failure = str(exc)
        self._collective_failure(failure)
        self._write(self.ψplus_q, ψplus)
        self._write(self.σ_q, σ)
        self._write(self.σplus_q, σplus)
        self._write(self.g_used_q, g)
        if hasattr(self, 'problem'):
            self.problem._constitutive_x = None



    def timestep(self, move_mesh=True):
        super().timestep(move_mesh=move_mesh)
        self.refresh_material_outputs()

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        self.refresh_material_outputs()

    def initial_infinite_shelf(self):
        super().initial_infinite_shelf()
        self.refresh_material_outputs()

    def initial_stokes(self, bc_func=None):
        super().initial_stokes(bc_func=bc_func)
        self.refresh_material_outputs()