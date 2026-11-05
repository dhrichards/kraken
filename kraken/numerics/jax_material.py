"""Experimental float64 no-tension material kernel, independent of dolfinx.

Mandel order: xx, yy, zz, sqrt(2)*yz, sqrt(2)*xz, sqrt(2)*xy.
Only first strain derivatives are needed: no AD through eigenvectors or Newton.
The exact spectral law replaces the old plane-strain eigenvalue regularisation.
"""
from functools import partial
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax

SQRT2 = jnp.sqrt(2.)
M = jnp.array([
    [[1,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,1]],
    [[0,0,0],[0,0,1],[0,1,0]],
    [[0,0,1],[0,0,0],[1,0,0]],
    [[0,1,0],[1,0,0],[0,0,0]],
], dtype=jnp.float64) / jnp.array([1,1,1,SQRT2,SQRT2,SQRT2])[:,None,None]
IDENTITY = jnp.array([1.,1.,1.,0.,0.,0.])
DEV = jnp.eye(6) - jnp.outer(IDENTITY, IDENTITY)/3
# Exactly the current dq=(q_xx,q_zz,q_xz) convention.
B2 = jnp.array([[1,0,0],[-1,-1,0],[0,1,0],[0,0,0],[0,0,SQRT2],[0,0,0]])
# General 3D traceless strain, adding yz and xy to the existing convention.
B3 = jnp.concatenate((B2, jnp.array([[0,0],[0,0],[0,0],[SQRT2,0],[0,0],[0,SQRT2]])), axis=1)


def to_mandel(tensor):
    return jnp.einsum('aij,...ij->...a', M, tensor)


def from_mandel(vector):
    return jnp.einsum('...a,aij->...ij', vector, M)


def free_energy_plus_nt_principal(λ, ν):
    """Single definition of the no-tension split; λ sorted ascending."""
    λ_μ = 2*ν/(1-2*ν)
    ψ1 = .5*λ_μ*jnp.sum(λ)**2 + jnp.dot(λ, λ)
    ψ2 = ψ1 - (1+ν)*λ[0]**2
    ψ3 = (1+ν)*((1-ν)*λ[2]+ν*(λ[0]+λ[1]))**2/((1-2*ν)*(1-ν**2))
    return jnp.where(λ[0] > 0, ψ1,
           jnp.where(λ[1]+ν*λ[0] > 0, ψ2,
           jnp.where((1-ν)*λ[2]+ν*(λ[0]+λ[1]) > 0, ψ3, 0.)))

_principal_grad = jax.grad(free_energy_plus_nt_principal, argnums=0)
_principal_hessian = jax.jacfwd(_principal_grad, argnums=0)


# Reduced plane-strain order: xx, yy, zz, sqrt(2)*xz.
PLANE = jnp.array([0, 1, 2, 4])
I4 = jnp.array([1., 1., 1., 0.])
DEV4 = jnp.eye(4)-jnp.outer(I4, I4)/3
B4 = B2[PLANE]
IXZ4 = jnp.diag(jnp.array([1., 0., 1., 1.]))
_principal_value_grad = jax.value_and_grad(free_energy_plus_nt_principal, argnums=0)


def _plane_spectrum(ε):
    """Direct projectors in four-component notation; no angles/eigenvectors."""
    mean = .5*(ε[0]+ε[2])
    diff = .5*(ε[0]-ε[2])
    shear = ε[3]/SQRT2
    radius = jnp.hypot(diff, shear)
    safe_radius = jnp.where(radius > 0, radius, 1.)
    # At exact degeneracy choose the x/z axes; the tangent uses the limit.
    a = jnp.where(radius > 0, diff/safe_radius, 1.)
    b = jnp.where(radius > 0, shear/safe_radius, 0.)
    Plo = jnp.array([.5*(1-a), 0., .5*(1+a), -b/SQRT2])
    Phi = jnp.array([.5*(1+a), 0., .5*(1-a), b/SQRT2])
    Pyy = jnp.array([0., 1., 0., 0.])
    λ = jnp.array([mean-radius, ε[1], mean+radius])
    order = jnp.where(λ[1] < λ[0], jnp.array([1,0,2]),
            jnp.where(λ[1] > λ[2], jnp.array([0,2,1]), jnp.array([0,1,2])))
    lo = jnp.where(λ[1] < λ[0], 1, 0)
    hi = jnp.where(λ[1] > λ[2], 1, 2)
    P = jnp.stack((Plo,Pyy,Phi))[order]
    return λ[order], P, Plo, Phi, radius, lo, hi


def eigenstate_plane_strain(ε):
    """Sorted eigenvalues and three 3x3 projectors for a plane-strain state."""
    λ, P, *_ = _plane_spectrum(ε[PLANE])
    P6 = jnp.zeros((3,6)).at[:,PLANE].set(P)
    return λ, from_mandel(P6)


def response_plus_nt_plane(ε, ν, tangent=True):
    """No-tension response in four components, with an optional 4x4 tangent.

    ε_yz and ε_xy are excluded, not approximated. Their tangent columns are
    unnecessary for this restricted kinematics. Only the xz pair rotates.
    """
    λ, P, Plo, Phi, radius, lo, hi = _plane_spectrum(ε)
    ψplus, σ_principal = _principal_value_grad(λ, ν)
    σplus = σ_principal @ P
    if not tangent:
        return ψplus, σplus, None
    H = _principal_hessian(λ, ν)
    gap = 2*radius
    close = gap <= 1e-10*jnp.maximum(jnp.max(jnp.abs(λ)), 1e-12)
    quotient = (σ_principal[hi]-σ_principal[lo])/jnp.where(close, 1., gap)
    limit = .5*(H[lo,lo]+H[hi,hi]-H[lo,hi]-H[hi,lo])
    quotient = jnp.where(close, limit, quotient)
    Cplus = P.T@H@P + quotient*(IXZ4-jnp.outer(Plo,Plo)-jnp.outer(Phi,Phi))
    return ψplus, σplus, Cplus


def elastic_response_plane(ε_e, g, p_crack, ν, tangent=True):
    λ_μ = 2*ν/(1-2*ν)
    ε_shifted = ε_e+p_crack/(3*(λ_μ+2/3))*I4
    ψplus, σplus, Cplus = response_plus_nt_plane(ε_shifted, ν, tangent=tangent)
    σ = 2*ε_e+λ_μ*jnp.dot(I4,ε_e)*I4+(g-1)*σplus
    C = (2*jnp.eye(4)+λ_μ*jnp.outer(I4,I4)+(g-1)*Cplus) if tangent else None
    return ψplus, σ, C, σplus


def local_system_plane(dq, ε_trial, data, tangent=True):
    g, p_crack, ν, dtstar, n, θ, fluidity_prev_it = data
    ψplus, σ, C, σplus = elastic_response_plane(ε_trial-B4@dq,g,p_crack,ν,tangent=tangent)
    s = DEV4@σ
    fluidity = jnp.where(n == 1, 1., (1-θ)*fluidity_prev_it+θ*jnp.dot(s,s)/2)
    r = B4.T@(B4@dq-dtstar*fluidity*s)
    if not tangent:
        return r
    θ_eff = jnp.where(n == 1, 0., θ)
    creep_tangent = fluidity*DEV4+θ_eff*jnp.outer(s,s)
    A = B4.T@(B4+dtstar*creep_tangent@C@B4)
    return r, A, ψplus, σ, C, creep_tangent, σplus


def response_plus_nt(ε, ν, plane_strain=False):
    """Energy, Mandel stress and tangent, with repeated-eigenvalue limits.

    With plane_strain=True, only the xx, yy, zz, xz tangent block is
    returned; excluded rows/columns are zero, not full 3D sensitivities.
    At a split switching surface the strict inequalities choose one branch.
    This is a piecewise tangent, not a claim of a classical Hessian there.
    """
    if plane_strain:
        ψplus, σplus, Cplus = response_plus_nt_plane(ε[PLANE], ν)
        σ6 = jnp.zeros(6).at[PLANE].set(σplus)
        C6 = jnp.zeros((6,6)).at[PLANE[:,None],PLANE[None,:]].set(Cplus)
        return ψplus, σ6, C6
    λ, V = jnp.linalg.eigh(from_mandel(ε))
    ψplus = free_energy_plus_nt_principal(λ, ν)
    σ_principal = _principal_grad(λ, ν)
    H = _principal_hessian(λ, ν)
    σ = to_mandel((V*σ_principal[None,:]) @ V.T)
    gap = λ[:,None] - λ[None,:]
    close = jnp.abs(gap) <= 1e-10*jnp.maximum(jnp.max(jnp.abs(λ)), 1e-12)
    safe_gap = jnp.where(close, 1., gap)
    quotient = (σ_principal[:,None]-σ_principal[None,:])/safe_gap
    # Symmetrised limit H_ii-H_ij for a permutation-invariant energy.
    limit = .5*(jnp.diag(H)[:,None]+jnp.diag(H)[None,:]-H-H.T)
    quotient = jnp.where(close, limit, quotient)
    local_basis = jnp.einsum('ki,akl,lj->aij', V, M, V)
    local_diag = jnp.einsum('ij,aj->ai', H, jnp.diagonal(local_basis, axis1=1, axis2=2))
    response = quotient[None,:,:]*local_basis
    idx = jnp.arange(3)
    response = response.at[:,idx,idx].set(local_diag)
    global_response = jnp.einsum('ki,aij,lj->akl', V, response, V)
    Cplus = jnp.einsum('bkl,akl->ba', M, global_response)
    return ψplus, σ, Cplus


def elastic_response(ε_e, g, p_crack, ν, plane_strain=False):
    λ_μ = 2*ν/(1-2*ν)
    C0 = 2*jnp.eye(6)+λ_μ*jnp.outer(IDENTITY, IDENTITY)
    ε_shifted = ε_e + p_crack/(3*(λ_μ+2/3))*IDENTITY
    ψplus, σplus, Cplus = response_plus_nt(ε_shifted, ν, plane_strain=plane_strain)
    return ψplus, C0 @ ε_e+(g-1)*σplus, C0+(g-1)*Cplus


def local_system(dq, ε_trial, data, B, plane_strain=False):
    # data: g, p_crack, ν, dtstar, n, θ, fluidity_prev_it
    g, p_crack, ν, dtstar, n, θ, fluidity_prev_it = data
    ψplus, σ, C = elastic_response(ε_trial-B@dq, g, p_crack, ν, plane_strain=plane_strain)
    s = DEV @ σ
    fluidity = jnp.where(n == 1, 1., (1-θ)*fluidity_prev_it+θ*jnp.dot(s,s)/2)
    θ_eff = jnp.where(n == 1, 0., θ)
    creep_tangent = fluidity*DEV + θ_eff*jnp.outer(s,s)
    r = B.T @ (B@dq-dtstar*fluidity*s)
    A = B.T @ (B+dtstar*creep_tangent@C@B)
    return r, A, ψplus, σ, C, creep_tangent


def make_integrator(dimension=2, max_it=50, max_backtracks=20, reduced_tangent=False, batch_size=None):
    """Build a batched CPU/GPU kernel. Inputs and tolerances stay dynamic.

    Returns dq, σ, ψplus, C_alg, residual/tolerance,
    success, Newton iterations, σplus. Failure is reported, never silently accepted.
    Supports current n=1 and n=3 laws only (validated by the adapter).
    In 2D all local tensor algebra is four-component. reduced_tangent=True
    returns a 4x4 tangent; otherwise that block is padded to 6x6 with zeros
    in excluded directions. Stress outputs remain six-component. batch_size
    optionally limits the number of points sharing a vectorised while loop.
    """
    if dimension not in (2,3):
        raise ValueError('dimension must be 2 or 3')
    B = B4 if dimension == 2 else B3
    plane_strain = dimension == 2
    system = local_system_plane if plane_strain else partial(local_system, B=B)
    residual = (partial(local_system_plane, tangent=False) if plane_strain
                else lambda dq, ε, data: system(dq, ε, data)[0])

    def point(ε_trial, data, guess, atol, rtol):
        valid_plane = jnp.array(True)
        if plane_strain:
            valid_plane = (ε_trial[3] == 0) & (ε_trial[5] == 0)
            ε_trial = ε_trial[PLANE]
        r0 = residual(jnp.zeros_like(guess), ε_trial, data)
        tol = atol+rtol*jnp.linalg.norm(r0)
        r = residual(guess, ε_trial, data)
        norm = jnp.linalg.norm(r)
        valid = jnp.isfinite(norm) & jnp.isfinite(tol) & (tol > 0)
        valid = valid & valid_plane

        def cond(state):
            dq, norm, it, ok = state
            return ok & (norm > tol) & (it < max_it)

        def step(state):
            dq, norm, it, ok = state
            r, A, *_ = system(dq, ε_trial, data)
            correction = jnp.linalg.solve(A, -r)

            def back_cond(bt):
                k, alpha, candidate, candidate_norm, accepted = bt
                return (~accepted) & (k <= max_backtracks)

            def back_step(bt):
                k, alpha, _, _, _ = bt
                candidate = dq+alpha*correction
                rn = jnp.linalg.norm(residual(candidate, ε_trial, data))
                accepted = jnp.isfinite(rn) & ((rn <= (1-1e-4*alpha)*norm) | (rn <= tol))
                return k+1, alpha*.5, candidate, rn, accepted

            _, _, candidate, rn, accepted = lax.while_loop(
                back_cond, back_step, (0, 1., dq, norm, jnp.array(False)))
            return candidate, rn, it+1, accepted

        dq, norm, it, ok = lax.while_loop(cond, step, (guess, norm, 0, valid))
        result = system(dq, ε_trial, data)
        r, A, ψplus, σ, C, creep_tangent = result[:6]
        # Implicit differentiation at the converged local solution.
        ddq_dε = jnp.linalg.solve(A, data[3]*B.T@creep_tangent@C)
        C_alg = C-C@B@ddq_dε
        ok = ok & (norm <= tol) & jnp.all(jnp.isfinite(C_alg)) & jnp.all(jnp.isfinite(σ)) & jnp.isfinite(ψplus)
        if plane_strain:
            σplus = result[6]
            σ = jnp.zeros(6).at[PLANE].set(σ)
            σplus = jnp.zeros(6).at[PLANE].set(σplus)
            if not reduced_tangent:
                C_alg = jnp.zeros((6,6)).at[PLANE[:,None],PLANE[None,:]].set(C_alg)
        else:
            λ_μ = 2*data[2]/(1-2*data[2])
            ε_shifted = ε_trial-B@dq+data[1]/(3*(λ_μ+2/3))*IDENTITY
            σplus = response_plus_nt(ε_shifted, data[2])[1]
        ok = ok & jnp.all(jnp.isfinite(σplus))
        return dq, σ, ψplus, C_alg, norm/tol, ok, it, σplus

    if batch_size is None:
        return jax.jit(jax.vmap(point, in_axes=(0,0,0,None,None)))
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer or None')

    # Optional bounded batches: limits how many easy points wait for a hard
    # point. Kept opt-in because it may lose CPU/GPU parallelism.
    def mapped(ε_trial, data, guess, atol, rtol):
        return lax.map(lambda args: point(*args, atol, rtol),
                       (ε_trial, data, guess), batch_size=batch_size)
    return jax.jit(mapped)


# Re-evaluate diagnostics at fixed internal strain; does not advance history.
def diagnostic_response(ε_e, g, p_crack, ν, plane_strain=False):
    if plane_strain:
        ψplus, σ, _, σplus = elastic_response_plane(ε_e[PLANE],g,p_crack,ν,tangent=False)
        return ψplus, jnp.zeros(6).at[PLANE].set(σ), jnp.zeros(6).at[PLANE].set(σplus)
    ψplus, σ, C = elastic_response(ε_e, g, p_crack, ν)
    λ_μ = 2*ν/(1-2*ν)
    σplus = response_plus_nt(ε_e+p_crack/(3*(λ_μ+2/3))*IDENTITY, ν, plane_strain=plane_strain)[1]
    return ψplus, σ, σplus


evaluate_batch = jax.jit(jax.vmap(diagnostic_response, in_axes=(0,0,0,0)))

# Static choice: the 2D diagnostic path also avoids the full eigensolver.
evaluate_batch_plane_strain = jax.jit(jax.vmap(
    partial(diagnostic_response, plane_strain=True), in_axes=(0,0,0,0)))