import ufl



def advection(φ, v, u, n, θ=1.0):
    """
    This form is called a_A in text

    u is the velocity
    φ is the first argument (Trial/Function)
    v is the second argument  (Test)

    n is the cell normal
    """

    un = abs(ufl.dot(u('+'), n('+')))

    a_cell = - θ*ufl.dot(u*φ, ufl.grad(v))*ufl.dx

    # Check this versus jump(v, n)
    # jump_v = v('+')*n('+') + v('-')*n('-')
    # jump_φ = φ('+')*n('+') + φ('-')*n('-')
    jump_v = ufl.jump(v, n)
    jump_φ = ufl.jump(φ, n)

    a_int = θ*(ufl.dot(u('+'), jump_v)*ufl.avg(φ) + 0.5*un*ufl.dot(jump_φ, jump_v))*ufl.dS
    a_ext = θ*ufl.dot(v, ufl.dot(u, n)*φ)*ufl.ds

    a = a_cell + a_int + a_ext
    return a

def diffusion(φ, v, kappa, alpha, n, h, θ=1.0):
    """
    This form is called a_D in text

    φ is the first argument (Trial/Function)
    v is the second argument  (Test)

    kappa is the diffusion _constant_

    alpha is the constant associated with the DG scheme
    n is the cell normal
    h is the cell size
    """

    # Contribution from the cells
    a_cell = θ*kappa*ufl.dot(ufl.grad(φ), ufl.grad(v))*ufl.dx

    # Contribution from the interior facets
    a_int0 = θ*kappa('+')*alpha('+')/h('+')*ufl.dot(ufl.jump(v, n), ufl.jump(φ, n))*ufl.dS
    a_int1 = - θ*kappa('+')*ufl.dot(ufl.avg(ufl.grad(v)), ufl.jump(φ, n))*ufl.dS
    a_int2 = - θ*kappa('+')*ufl.dot(ufl.jump(v, n), ufl.avg(ufl.grad(φ)))*ufl.dS
    a_int = a_int0 + a_int1 + a_int2

    # Contribution from the exterior facets?

    a = a_cell + a_int
    return a

def source(v, qs):
    """
    Corresponds to source term in text

    v is Test function

    qs is source function
    """

    # Contribution from the cells
    a = -v*qs*ufl.dx

    return a


def backward_euler(u, u_, dt):
    return (u - u_)/dt