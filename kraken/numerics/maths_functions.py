import ufl
from .invariants import eigenstate
from dolfinx import fem, default_scalar_type
import numpy as np

def viscous_strain(q):
    return ufl.as_tensor([
        [q[0], 0,             q[2]],
        [0,   -q[0] - q[1],   0],
        [q[2], 0,             q[1]],
    ])

def L(du):
    # Mesh coordinates: (x, z)
    # Tensor coordinates: (x, y, z)
    return ufl.as_tensor([
        [du[0].dx(0), 0, du[0].dx(1)],
        [0,             0, 0],
        [du[1].dx(0), 0, du[1].dx(1)],
    ])


def tr_e(A):
    ''' In 3D normal trace, in 2D a special trace, assuming that we have plane viscous strain rates,
      then there is out of plane elastic strain which is always equal to:
      e_yy = (e_xx + e_zz)/2
      so that the trace is:
      tr(A) = e_xx + e_yy + e_zz = 1.5*(e_xx + e_zz)'''
    if ufl.shape(A)[0] == 2:
        return 1.5*ufl.tr(A)
    else:
        return ufl.tr(A)

def viscosity(ε, n, eps=1.e-11, A=1.0): 
    '''Viscosity from Glens law'''
    εe2 = ufl.inner(ε, ε) / 2 + eps
    return  A**(-1/n) * εe2**((1 - n) / (2 * n))

def viscosity_stress(σD,n,eps=1e-11,A=1.0):
    σDe2 = ufl.inner(σD,σD)/2 + eps
    return (1/A)*σDe2**((1 - n) / 2)

def viscous_energy(ε, n, eps = 1e-12, A=1.0):
    η = viscosity(ε, n, eps, A)
    return (2*n/(n+1))*η*ufl.inner(ε, ε) 


def ε(u):
    return ufl.sym(ufl.grad(u))

def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def positive_part(x):
    return ufl.max_value(0.0,x)


def negative_part(x):
    return 0.5*(x-abs(x))


def water_pressure(msh,v,ρw,ucstar=1.0, level = 0.0):
    '''Water pressure, including how displacement changes water pressure'''
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + ucstar*v[msh.geometry.dim-1]
    z = z - level
    return ufl.max_value(0.0,-ρw*z) 

def water_pressure_static(msh,ρw,level=0.0):
    '''Water pressure, not dependent on displacement'''
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    z = z - level
    return ufl.max_value(0.0,-ρw*z) 

def modified_water_pressure(msh,ρw,ρm,sealevel=0.0,cracklevel=0.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    pw = ufl.max_value(0.0,-ρw*(z-sealevel))
    pc = ufl.max_value(0.0,-ρw*(z-cracklevel))


    def smoothstep(x, x_c, width):
        return 0.5*(1 + ufl.tanh((x-x_c)/width))

    def smoothtransition(a, b, x, x_c, width):
        return a + (b-a)*smoothstep(x, x_c, width)
    # return ufl.max_value(pw,pc)
    return smoothtransition(pw,pc, z, 0.7, 0.05)




def body_force(msh):
    '''Body force'''
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0.0, -1.0)))
    else:
        f = fem.Constant(msh, default_scalar_type((0.0, 0.0,-1.0)))
    return f


def rate_factor(T):
    '''Ice Rate factor in ufl (T in celcius)'''
    # https://elmerice.elmerfem.org/wiki/doku.php?id=problems:rheology
    R = 8.314
    Q = ufl.conditional(ufl.gt(T,-10),115e3,60e3)
    A0 = ufl.conditional(ufl.gt(T,-10),2.42736e-02,2.89165e-13)
    # Q = 60e3
    # A0 = 2.89165e-13
    return A0*ufl.exp(-Q/(R*(T+273.15)))

def rate_factor_np(T):
    '''Ice rate factor in numpy (T in celcius)'''
    # https://elmerice.elmerfem.org/wiki/doku.php?id=problems:rheology
    R = 8.314
    Q = np.where(T>-10,115e3,60e3)
    A0 = np.where(T>-10,2.42736e-02,2.89165e-13)
    # Q = 60e3
    # A0 = 2.89165e-13
    return A0*np.exp(-Q/(R*(T+273.15)))


def corotational_increment(du, ucstar):
    """
    Returns:
        R3: incremental rotation, embedded in 3D
        de3: logarithmic stretch increment in the new spatial frame,
             scaled consistently with sym(L(du))

    Assumes:
        - tensor ordering is x, y, z, with y out of plane;
        - det(f2) > 0;
        - ucstar > 0.
    """
    L3 = L(du)

    L2 = ufl.as_matrix((
        (L3[0, 0], L3[0, 2]),
        (L3[2, 0], L3[2, 2]),
    ))

    I2 = ufl.Identity(2)
    f2 = I2 + ucstar * L2

    # Proper polar rotation: f2 = V2 * R2.
    c = f2[0, 0] + f2[1, 1]
    s = f2[1, 0] - f2[0, 1]
    norm = ufl.sqrt(c*c + s*s)

    R2 = ufl.as_matrix((
        (c, -s),
        (s,  c),
    )) / norm

    V2 = ufl.sym(ufl.dot(f2, R2.T))

    # Matrix logarithm of the symmetric positive-definite V2.
    m = 0.5 * ufl.tr(V2)
    B = V2 - m * I2

    # V2 has eigenvalues m * (1 +/- t).
    t2 = 0.5 * ufl.inner(B, B) / (m*m)

    # Safe evaluation at repeated eigenvalues.
    t = ufl.sqrt(ufl.max_value(t2, 1.0e-16))

    ratio = ufl.conditional(
        ufl.lt(t2, 1.0e-8),
        1.0 + t2 / 3.0 + t2*t2 / 5.0,
        0.5 * ufl.ln((1.0 + t) / (1.0 - t)) / t,
    )

    logV2 = (
        0.5 * ufl.ln(ufl.det(V2)) * I2
        + (ratio / m) * B
    )

    de2 = logV2 / ucstar

    R3 = ufl.as_matrix((
        (R2[0, 0], 0.0, R2[0, 1]),
        (0.0,      1.0, 0.0),
        (R2[1, 0], 0.0, R2[1, 1]),
    ))

    de3 = ufl.as_matrix((
        (de2[0, 0], 0.0, de2[0, 1]),
        (0.0,       0.0, 0.0),
        (de2[1, 0], 0.0, de2[1, 1]),
    ))

    return R3, de3

