import ufl
from .invariants import eigenstate
from dolfinx import fem, default_scalar_type
import numpy as np

def tr_e(A):
    ''' Special trace, assuming that we have plane viscous strain rates,
      then there is out of plane elastic strain which is always equal to:
      e_yy = (e_xx + e_zz)/2
      so that the trace is:
      tr(A) = e_xx + e_yy + e_zz = 1.5*(e_xx + e_zz)'''
    if ufl.shape(A)[0] == 2:
        return 1.5*ufl.tr(A)
    else:
        return ufl.tr(A)

def viscosity(ε, n, eps=1.e-11, A=1.0): 
    εe2 = ufl.inner(ε, ε) / 2 + eps
    return  A**(-1/n) * εe2**((1 - n) / (2 * n))

def viscous_energy(ε, n, eps = 1e-12, A=1.0):
    η = viscosity(ε, n, eps, A)
    return (2*n/(n+1))*η*ufl.inner(ε, ε) 


def ε(u):
    return ufl.sym(ufl.grad(u))

def dev3(A):
    δ = ufl.Identity(ufl.shape(A)[0])
    return A - ufl.tr(A)/3*δ

def εD3(u):
    return dev3(ε(u))


def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def positive_part(x,eps=1e-8):
    return ufl.max_value(0.0,x)


def negative_part(x,eps=1e-6):
    return 0.5*(x-abs(x))


def water_pressure(msh,v,ρw,ucstar=1.0, level = 0.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + ucstar*v[msh.geometry.dim-1]
    z = z - level
    return ufl.max_value(0.0,-ρw*z) 

def water_pressure_static(msh,ρw,level=0.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    z = z - level
    return ufl.max_value(0.0,-ρw*z) 


def body_force(msh):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0.0, -1.0)))
    else:
        f = fem.Constant(msh, default_scalar_type((0.0, 0.0,-1.0)))
    return f


def rate_factor(T):
    # https://elmerice.elmerfem.org/wiki/doku.php?id=problems:rheology
    R = 8.314
    Q = ufl.conditional(ufl.gt(T,-10),115e3,60e3)
    A0 = ufl.conditional(ufl.gt(T,-10),2.42736e-02,2.89165e-13)
    # Q = 60e3
    # A0 = 2.89165e-13
    return A0*ufl.exp(-Q/(R*(T+273.15)))

def rate_factor_np(T):
    # https://elmerice.elmerfem.org/wiki/doku.php?id=problems:rheology
    R = 8.314
    Q = np.where(T>-10,115e3,60e3)
    A0 = np.where(T>-10,2.42736e-02,2.89165e-13)
    # Q = 60e3
    # A0 = 2.89165e-13
    return A0*np.exp(-Q/(R*(T+273.15)))

