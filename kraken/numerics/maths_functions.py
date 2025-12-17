import ufl
from .invariants import eigenstate
from dolfinx import fem, default_scalar_type
import numpy as np

def viscosity(ε, n, eps=1.e-11, A=1.0): 
    εe2 = ufl.inner(ε, ε) / 2 + eps
    return  A**(-1/n) * εe2**((1 - n) / (2 * n))

def viscosity_stress(σ, n, eps=1.e-14, A=1.0):
    σD = dev3(σ)
    σDe2 = ufl.inner(σD, σD) / 2 + eps
    return A**(-1)* σDe2**((1 - n) / 2)


def viscous_energy(ε, n, eps = 1e-12, A=1.0):
    η = viscosity(ε, n, eps, A)
    return (2*n/(n+1))*η*ufl.inner(ε, ε) 


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


def temperature(msh,ρistar,Ts=-20.0,Tb=-2.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    δ = 1 - ρistar
    z = z - δ
    #linear profile
    T = -(Tb - Ts)*z + Ts
    return T


def tensor_2d_to_3d(A2):
    return ufl.as_tensor([[A2[0,0], A2[0,1], 0.0],
                          [A2[1,0], A2[1,1], 0.0],
                          [0.0,      0.0,    0.0]])

def tensor_3d_to_2d(A3):
    return ufl.as_tensor([[A3[0,0], A3[0,1]],
                          [A3[1,0], A3[1,1]]])


def grad3d(u):
    gradu = ufl.grad(u)
    return ufl.as_tensor([[gradu[0,0], gradu[0,1], 0],
                                [gradu[1,0], gradu[1,1], 0],
                                [gradu[2,0], gradu[2,1], 0]])

def ε3d(u):
    return 0.5*(grad3d(u) + ufl.transpose(grad3d(u)))



def v2to3(v):
    return ufl.as_vector([v[0], v[1], 0])

def v3to2(v):
    return ufl.as_vector([v[0], v[1]])


def deviatoric2d_to_3d(A):
    return ufl.as_tensor([[A[0,0], A[0,1], 0.0],
                          [A[1,0], A[1,1], 0.0],
                          [0.0,      0.0,   -A[0,0]-A[1,1]]])


def short_voigt2tensor(v):
    return ufl.as_tensor([[v[0], v[2], 0.0],
                          [v[2], v[1], 0.0],
                          [0.0,   0.0,  v[3]]])


def tensor2short_voigt(A):
    return ufl.as_vector([A[0,0], A[1,1], A[2,2], A[0,1]])