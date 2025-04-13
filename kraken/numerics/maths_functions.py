import ufl
from .invariants import matrix_function, eigenstate
from . import energy_splits as es
from dolfinx import fem, default_scalar_type

def viscosity(u, n, eps=1.e-8, A=1.0): 
    return 0.5* A**(-1/n) * (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))

def ε(u):
    return ufl.sym(ufl.grad(u))

def cauchy_stress(ε,λ,μ):
    D = ufl.shape(ε)[0]
    return λ*ufl.tr(ε)*ufl.Identity(D) + 2*μ*ε 

def viscous_stress(u,p,η):
    δ = ufl.Identity(len(u))
    return -p*δ + 2*η(u)*ε(u)


def largest_eigenvalue(A):
    λ, M = eigenstate(A)
    return λ[-1]

def principal_stress(ε,λ,μ):
    return largest_eigenvalue(cauchy_stress(ε,λ,μ))



def degradation_default(d,k=1e-5):
    return (1-k)*(1-d)**2 + k


def degradation_Lo2023(d,q=1.0):
    ϕ = 1-d
    return (q+1)*(1 - (q/(q+1))**(ϕ**2) )

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw




def degraded_free_energy(ε,g,λ,μ,ψcrit,free_energy_plus=es.free_energy_plus_spectral):
    ψplus = (free_energy_plus(ε,λ,μ)-ψcrit)
    # # ψplus = free_energy_plus(u,ν)
    ψminus = es.free_energy(ε,λ,μ) - ψplus
    return g*ψplus + ψminus



def history_function(ε,Hprev,λ,μ,ψcrit,free_energy_plus=es.free_energy_plus_spectral):
    ψp = free_energy_plus(ε,λ,μ) - ψcrit
    return ufl.max_value(ψp,Hprev)
    # return ufl.conditional(ufl.gt(ψp,Hprev),ψp,Hprev)


def water_pressure(msh,vh,ρw,g,patm):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + vh[msh.geometry.dim-1]
    return ufl.max_value(0.0,-ρw*g*z) + patm
    # return ufl.conditional(ufl.gt(z,0),0.0,-ρw*g*z)
    # return 0.5*(-z + (z**2 + eps**2)**0.5)


def body_force(msh,ρi,g, α=0.0):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((
                        ρi*g*ufl.sin(α*ufl.pi/180), 
                        -ρi*g*ufl.cos(α*ufl.pi/180))))
    else:
        f = fem.Constant(msh, default_scalar_type((
            ρi*g*ufl.sin(α*ufl.pi/180),
            0.0,
            -ρi*g*ufl.cos(α*ufl.pi/180))))
    return f
