import ufl
from .invariants import matrix_function, eigenstate
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

# def deriv_deg_wrt_damage(d,k=1e-5):
#     return -2*(1-d) + k

# def degradation(d,s=200):
#     ϕ = 1-d
#     return s*(1 - ((s-1)/s)**(ϕ**2) )

# def deriv_deg_wrt_damage(d,dlin,s=200):
#     ϕ = 1-dlin
#     return -2*s*(1-d)*((s-1)/s)**(ϕ**2)*ufl.ln((s-1)/s)

def γ(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw


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
    σminus = σ - σplus
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


def projection_split(ε,ε_prev):

    i,j,k,l = ufl.indices(4)
    Pplus = projection_tensor(ε_prev)

    εplus = ufl.as_tensor(Pplus[i,j,k,l]*ε[k,l],(i,j))
    εminus = ε - εplus

    return εplus,εminus
    


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








     