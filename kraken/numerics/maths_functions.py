import ufl
from .invariants import eigenstate
from . import energy_splits as es
from dolfinx import fem, default_scalar_type

def viscosity(ε, n, eps=1.e-11, A=1.0): 
    return 0.5* A**(-1/n) * (ufl.inner(ε, ε) / 2 + eps)**((1 - n) / (2 * n))

def viscous_stress(ε,p,η,C2):
    D = ufl.shape(ε)[0]
    δ = ufl.Identity(D)
    return η*ε/C2 - p*δ

def ε(u):
    return ufl.sym(ufl.grad(u))

def εD(u):
    return ufl.dev(ε(u))


def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def principal_stress(ε,λ,μ):
    return largest_eigenvalue(es.cauchy_stress(ε,λ,μ))


def degradation_default(d,k=1e-5):
    return (1-k)*(1-d)**2 + k


def degradation_Lo2023(d,q=1.0,k=1e-5):
    ϕ = 1-d
    g = (q+1)*(1 - (q/(q+1))**(ϕ**2) )
    return (1-k)*g + k

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw




def degraded_free_energy(ε,g,ν,ψcrit,free_energy_plus=es.free_energy_plus_spectral):
    ψplus = (free_energy_plus(ε,ν)-ψcrit)
    # # ψplus = free_energy_plus(u,ν)
    ψminus = es.free_energy(ε,ν) - ψplus
    return g*ψplus + ψminus



def history_function(ε,Hprev,ν,ψcrit,free_energy_plus=es.free_energy_plus_spectral):
    ψp = es.free_energy_plus_spectral(ε,ν) - ψcrit
    return ufl.max_value(ψp,Hprev)
    # return ufl.conditional(ufl.gt(ψp,Hprev),ψp,Hprev)


def water_pressure(msh,v,ucstar=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + ucstar*v[msh.geometry.dim-1]
    return ufl.max_value(0.0,-z) 
    # return ufl.conditional(ufl.gt(z,0),0.0,-ρw*g*z)
    # return 0.5*(-z + (z**2 + eps**2)**0.5)

def water_pressure_static(msh):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    return ufl.max_value(0.0,-z) 
    # return ufl.conditional(ufl.gt(z,0),0.0,-ρw*g*z)
    # return 0.5*(-z + (z**2 + eps**2)**0.5)


def body_force(msh,ρistar, α=0.0):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((
                        ρistar*ufl.sin(α*ufl.pi/180), 
                        -ρistar*ufl.cos(α*ufl.pi/180))))
    else:
        f = fem.Constant(msh, default_scalar_type((
            ρistar*ufl.sin(α*ufl.pi/180),
            0.0,
            -ρistar*ufl.cos(α*ufl.pi/180))))
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


def overburden_pressure(msh,ρistar, u, ucstar=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + ucstar*u[msh.geometry.dim-1]
    δ = 1 - ρistar
    zi = z - δ
    return -ρistar*zi


def clayton_driving_function(σ, σ_crit,pw=0.0):
    λ,_ = eigenstate(σ)
    Dd = 0.0
    for σa in λ:
        Dd += (es.positive_part(σa+pw)/σ_crit)**2 - 1.0

    return es.positive_part(Dd)


def ice_density(msh,ρistar_bottom=0.9, ρistar_top=0.4):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    


    ρi = ρistar_bottom + (ρistar_top - ρistar_bottom) * ufl.exp(-z)
    return ρi

