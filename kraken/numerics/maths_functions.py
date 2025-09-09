import ufl
from .invariants import eigenstate
from dolfinx import fem, default_scalar_type
import numpy as np

def viscosity(ε, n, eps=1.e-11, A=1.0): 
    εe2 = ufl.inner(ε, ε) / 2 + eps
    return  A**(-1/n) * εe2**((1 - n) / (2 * n))

def viscosity_stress(σD, n, eps=1.e-11, A=1.0):
    τe2 = ufl.inner(σD, σD) / 2 + eps
    return A**(-1)* τe2**((1 - n) / 2)


def viscous_energy(ε, n):
    η = viscosity(ε, n, 1.e-8)
    return (n/(n+1))*η*ufl.inner(ε, ε) 


def viscous_stress(ε,p,η,C2=1):
    D = ufl.shape(ε)[0]
    δ = ufl.Identity(D)
    return η*ε/C2 - p*δ

def ε(u):
    return ufl.sym(ufl.grad(u))

def dev3(A):
    δ = ufl.Identity(ufl.shape(A)[0])
    return A - ufl.tr(A)/3*δ

def εD(u):
    return dev3(ε(u))


def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def positive_part(x,eps=1e-8):
    # return 0.5*(x + (x**2 + eps**2)**0.5)
    return ufl.max_value(0.0,x)
    # return ufl.conditional(ufl.gt(x,0),x,0)
    # return 0.5*(x + abs(x))
    # return 0.5*(x + ufl.sign(x)*x)


def negative_part(x,eps=1e-6):
    return 0.5*(x-abs(x))
    # return 0.5*(x - (x**2 + eps**2)**0.5)


def water_pressure(msh,v,ucstar=1.0, level = 0.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + ucstar*v[msh.geometry.dim-1]
    z = z - level
    return ufl.max_value(0.0,-z) 
    # return ufl.conditional(ufl.gt(z,0),0.0,-ρw*g*z)
    # return 0.5*(-z + (z**2 + eps**2)**0.5)

def water_pressure_static(msh,level=0.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    z = z - level
    return ufl.max_value(0.0,-z) 
    # return ufl.conditional(ufl.gt(z,level),0.0,-ρw*g*z)
    # return 0.5*(-z + (z**2 + eps**2)**0.5)


def body_force(msh):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0.0, -1.0)))
    else:
        f = fem.Constant(msh, default_scalar_type((0.0, 0.0,-1.0)))
    return f

def external_density(msh):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    ρa = 1e-3; ρw = 1.0
    return ufl.conditional(ufl.gt(z,0),0.0,ρw)


def water_body_force(msh,α=0.0):
    ρw = external_density(msh)
    if msh.geometry.dim == 2:
        f = ρw*fem.Constant(msh, default_scalar_type((
                        ufl.sin(α*ufl.pi/180), 
                        -ufl.cos(α*ufl.pi/180))))
    else:
        f = ρw*fem.Constant(msh, default_scalar_type((
            ufl.sin(α*ufl.pi/180),
            0.0,
            -ufl.cos(α*ufl.pi/180))))
    return f



def body_force_with_water(msh,ρi,g):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    f = fem.Constant(msh, default_scalar_type((0.0,-1.0)))

    ρa = 1e-3; ρw = 1.0
    ρw = ufl.conditional(ufl.gt(z,0),ρa,ρw)

    return g*ρi*f + (1-g)*ρw*f


def overburden_pressure(msh,ρistar, uz=0.0, ucstar=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + ucstar*uz
    δ = 1 - ρistar
    zi = z - δ
    return -ρistar*zi




def ice_density(msh,ρistar,ρfstar,Dstar):
    x = ufl.SpatialCoordinate(msh)

    Hwstar = flotation_height(ρistar,ρfstar,Dstar)
    z = x[msh.geometry.dim-1] + Hwstar

    
    return ρistar - (ρistar - ρfstar)*ufl.exp(-(1.0-z)/Dstar)

def ice_E(msh,params):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]

    E0 = 1.0
    Ef = 1.5/params.E
    D = 32.5/params.L

    return E0 - (E0 - Ef)*ufl.exp(-(1-z)/D)



def flotation_height(ρistar,ρfstar,Dstar):
    return ρistar - (ρistar - ρfstar)*(Dstar - Dstar*np.exp(-1/Dstar))
