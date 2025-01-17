import ufl
from invariants import matrix_function
from dolfinx import fem, default_scalar_type

def viscosity(u, n, eps=1.e-8, A=1.0): 
    return A**(-1/n) * (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))

def ε(u):
    return ufl.sym(ufl.grad(u))

def stress(u,ν):
    λoverμ = 2*ν/(1-2*ν)
    return λoverμ*ufl.tr(ε(u))*ufl.Identity(len(u)) + 2*ε(u)


def free_energy(ε,ν):
    λoverμ = 2*ν/(1-2*ν)
    return 0.5*λoverμ*ufl.tr(ε)**2 + ufl.inner(ε,ε)


def positive_part(x):
    return ufl.max_value(x,0)


def degradation(d,k=1e-5):
    return (1-d)**2 + k


def γ(d,l):
    return 0.5/l * (d**2 + l**2 * ufl.inner(ufl.grad(d), ufl.grad(d)))


def free_energy_plus(ε,ν):
# based on alternative formulation, equivalent to below
    λoverμ = 2*ν/(1-2*ν)
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)


def degraded_free_energy(ε,d,ν,ψcritstar):
    ψplus = free_energy_plus(ε,ν)-ψcritstar
    # ψplus = free_energy_plus(u,ν)
    ψminus = free_energy(ε,ν) - ψplus
    return degradation(d)*(ψplus) + (ψminus)


def degraded_stress(u,d,ν):
    λoverμ = 2*ν/(1-2*ν); I = ufl.Identity(len(u))
    σ = λoverμ*ufl.tr(ε(u))*I + 2*ε(u)   
    σplus = λoverμ*positive_part(ufl.tr(ε(u)))*I + \
        2*matrix_function(ε(u),positive_part)
    σminus = λoverμ*negative_part(ufl.tr(ε(u)))*I + \
        2*matrix_function(ε(u),negative_part)

    return degradation(d)*σplus + σminus

def degraded_pressure(p,d):
    pplus = positive_part(p)
    pminus = p - pplus
    return degradation(d)*pplus + pminus


def history_function(ε,Hprev,ν,ψcrit):
    ψp = free_energy_plus(ε,ν) - ψcrit
    return ufl.max_value(ψp,Hprev)


def water_pressure_static(msh,ρw=1.0,g=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    pw = ufl.conditional(ufl.lt(z, 0),
                         -ρw*g*z,
                         0.0)
    return pw


def water_pressure(msh,vh):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + vh[msh.geometry.dim-1]#*material.uc/material.L
    return ufl.max_value(0.0,-z)


def body_force(msh,ρratio):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))
    return f





     