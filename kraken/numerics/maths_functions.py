import ufl
from .invariants import matrix_function, eigenstate
from dolfinx import fem, default_scalar_type

def viscosity(u, n, eps=1.e-8, A=1.0): 
    return 0.5* A**(-1/n) * (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))

def ε(u):
    return ufl.sym(ufl.grad(u))

def cauchy_stress(ε,ν):
    λoverμ = 2*ν/(1-2*ν)
    D = ufl.shape(ε)[0]
    return λoverμ*ufl.tr(ε)*ufl.Identity(D) + 2*ε

def viscous_stress(u,p,η):
    δ = ufl.Identity(len(u))
    return -p*δ + 2*η(u)*ε(u)


def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def principal_stress(ε,ν):
    return largest_eigenvalue(cauchy_stress(ε,ν))



def free_energy(ε,ν):
    λoverμ = 2*ν/(1-2*ν)
    return 0.5*λoverμ*ufl.tr(ε)**2 + ufl.inner(ε,ε)


def positive_part(x,eps=1e-12):
    # return 0.5*(x + (x**2 + eps**2)**0.5)
    return ufl.max_value(0.0,x)


# def negative_part(x,eps=1e-6):
#     return 0.5*(x - (x**2 + eps**2)**0.5)



def degradation_default(d,k=1e-5):
    return (1-d)**2 + k


def degradation_Lo2023(d,q=1.0):
    ϕ = 1-d
    return (q+1)*(1 - (q/(q+1))**(ϕ**2) )

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw

def free_energy_plus(ε,ν):
    λoverμ = 2*ν/(1-2*ν)
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)


def free_energy_plus_amor(ε,ν):
    λoverμ = 2*ν/(1-2*ν)
    D = ufl.shape(ε)[0]
    return 0.5*(λoverμ+2/D)*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(ufl.dev(ε),ufl.dev(ε))


def degraded_free_energy(ε,g,ν,ψcritstar):
    ψplus = free_energy_plus(ε,ν)-ψcritstar
    # ψplus = free_energy_plus(u,ν)
    ψminus = free_energy(ε,ν) - ψplus
    return g*(ψplus) + (ψminus)


def degraded_stress(ε,g,ν):
    D = ufl.shape(ε)[0]
    λoverμ = 2*ν/(1-2*ν); I = ufl.Identity(D)
    σ = λoverμ*ufl.tr(ε)*I + 2*ε  
    σplus = λoverμ*positive_part(ufl.tr(ε))*I + \
        2*matrix_function(ε,positive_part)
    σminus = σ - σplus
    return g*σplus + σminus

def degraded_pressure(p,g):
    pplus = positive_part(p)
    pminus = p - pplus
    return g*pplus + pminus

def history_function(ε,Hprev,ν,ψcrit):
    ψp = free_energy_plus(ε,ν) - ψcrit
    # ψp = free_energy(ε,ν)
    return ufl.max_value(ψp,Hprev)


def water_pressure_static(msh,ρw=1.0,g=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    pw = ufl.conditional(ufl.lt(z, 0),
                         -ρw*g*z,
                         0.0)
    return pw


def water_pressure(msh,vh,uc_star=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + vh[msh.geometry.dim-1]*uc_star
    return ufl.max_value(0.0,-z)

def body_force(msh,ρratio, α=0.0):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((
                        ρratio*ufl.sin(α*ufl.pi/180), 
                        -ρratio*ufl.cos(α*ufl.pi/180))))
    else:
        f = fem.Constant(msh, default_scalar_type((
            ρratio*ufl.sin(α*ufl.pi/180),
            0.0,
            -ρratio*ufl.cos(α*ufl.pi/180))))
    return f
