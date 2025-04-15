import ufl
from .invariants import eigenstate
from . import energy_splits as es
from dolfinx import fem, default_scalar_type

def viscosity(u, n, eps=1.e-8, A=1.0): 
    return 0.5* A**(-1/n) * (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))

def viscous_stress(u,p,η):
    δ = ufl.Identity(len(u))
    return -p*δ + 2*η(u)*ε(u)

def ε(u):
    return ufl.sym(ufl.grad(u))





def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def principal_stress(ε,λ,μ):
    return largest_eigenvalue(es.cauchy_stress(ε,λ,μ))


def degradation_default(d,k=1e-5):
    return (1-k)*(1-d)**2 + k


def degradation_Lo2023(d,q=1.0):
    ϕ = 1-d
    return (q+1)*(1 - (q/(q+1))**(ϕ**2) )

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw




def degraded_free_energy(ε,g,ν,ψcrit,free_energy_plus=es.free_energy_plus_spectral):
    ψplus = (free_energy_plus(ε,ν)-ψcrit)
    # # ψplus = free_energy_plus(u,ν)
    ψminus = es.free_energy(ε,ν) - ψplus
    return g*ψplus + ψminus



def history_function(ε,Hprev,ν,ψcrit,free_energy_plus=es.free_energy_plus_spectral):
    ψp = free_energy_plus(ε,ν) - ψcrit
    return ufl.max_value(ψp,Hprev)
    # return ufl.conditional(ufl.gt(ψp,Hprev),ψp,Hprev)


def water_pressure(msh,v):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + v[msh.geometry.dim-1]
    return ufl.max_value(0.0,-z) 
    # return ufl.conditional(ufl.gt(z,0),0.0,-ρw*g*z)
    # return 0.5*(-z + (z**2 + eps**2)**0.5)


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
