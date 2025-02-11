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

def free_energy_alt(ε,ν):
    # use formulation in preprint that I think is wrong
    λoverμ = 2*ν/(1-2*ν)
    D = ufl.shape(ε)[0]
    return 0.5*(λoverμ + 2/D)*ufl.tr(ε)**2 + ufl.inner(ε,ε)


def positive_part(x):
    return ufl.max_value(x,0)


def degradation_default(d,k=1e-5):
    return (1-d)**2 + k

# def deriv_deg_wrt_damage(d,dlin):
#     return -2*(1-d)

def degradation_Lo2023(d,q=1.0):
    ϕ = 1-d
    return (q+1)*(1 - (q/(q+1))**(ϕ**2) )

# def deriv_deg_wrt_damage(d,dlin,q=200):
#     ϕ = 1-dlin
#     a = q/(q+1)
#     # return 2*(q+1)*(1-d)*ufl.ln(a)*(1+ϕ**2*ufl.ln(a)) # last term is taylor series of a**ϕ**2
#     return 2*(q+1)*(1-d)*ufl.ln(a)*a**(ϕ**2)

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw


def free_energy_plus(ε,ν):

    λoverμ = 2*ν/(1-2*ν)
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)

def free_energy_plus_alt(ε,ν):
    # use formulation in preprint that I think is wrong
    λoverμ = 2*ν/(1-2*ν)
    D = ufl.shape(ε)[0]
    εDplus = matrix_function(ufl.dev(ε),positive_part)
    return 0.5*(λoverμ + 2/D)*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εDplus,εDplus)





def free_energy_plus_P(ε,ε_prev,ν):
    i,j,k,l = ufl.indices(4)
    λoverμ = 2*ν/(1-2*ν)
    P = projection_tensor(ε_prev)
    εplus = ufl.as_tensor(P[i,j,k,l]*ε[k,l],(i,j))
    return 0.5*λoverμ*(heaviside(ufl.tr(ε_prev))*ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)


def degraded_free_energy(ε,g,ν,ψcritstar):
    ψplus = free_energy_plus(ε,ν)-ψcritstar
    # ψplus = free_energy_plus(u,ν)
    ψminus = free_energy(ε,ν) - ψplus
    return g*(ψplus) + (ψminus)

def degraded_free_energy_alt(ε,g,ν,ψcritstar):
    ψplus = free_energy_plus_alt(ε,ν)-ψcritstar
    # ψplus = free_energy_plus(u,ν)
    ψminus = free_energy_alt(ε,ν) - ψplus
    return g*(ψplus) + (ψminus)


def degraded_free_energy_P(ε,ε_prev,g,ν,ψcritstar):
    ψplus = free_energy_plus_P(ε,ε_prev,ν)-ψcritstar
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


def heaviside(x):
    return ufl.conditional(ufl.gt(x,0),1,0)


def degraded_stress_P(u,u_prev,g,ν):
    λoverμ = 2*ν/(1-2*ν); I = ufl.Identity(len(u))
    i,j,k,l = ufl.indices(4)
    σ = λoverμ*ufl.tr(ε(u))*I + 2*ε(u)
    P = projection_tensor(ε(u_prev))
    σplus = λoverμ*heaviside(ufl.tr(ε(u_prev)))*ufl.tr(ε(u))*I + \
        2*ufl.as_tensor(P[i,j,k,l]*ε(u)[k,l],(i,j))
    σminus = σ - σplus
    return g*σplus + σminus

    


def projection_tensor(ε):
    D = ufl.shape(ε)[0]
    λ, M = eigenstate(ε)

    if D == 3:
        bofa = [[1,2],[0,2],[0,1]]
    elif D == 2:
        bofa = [[1],[0]]

    P = ufl.zero((D,D,D,D))

    for a in range(D):
        P += heaviside(positive_part(λ[a]))*ufl.outer(M[a],M[a])


    for a in range(D):
        for b in bofa[a]:
            P += 0.5*θab(λ[a],λ[b])*tensor_commuter(M[a],M[b])

    return P

def θab(λa,λb):
    return (positive_part(λa)-positive_part(λb))/(λa-λb)


def tensor_commuter(A,B):
    i,j,k,l = ufl.indices(4)
    return ufl.as_tensor(A[i,k]*B[j,l] + A[i,l]*B[j,k],(i,j,k,l))








     