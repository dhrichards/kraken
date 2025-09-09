import ufl
from .invariants import matrix_function, eigenstate
from .maths_functions import ε, εD
from .energy_splits import positive_part



def heaviside(x,eps=1e-8):
    # return ufl.conditional(ufl.gt(x,0),1,0)
    return 0.5*(ufl.sign(x)+1)
    # return 0.5*(1 + x/(x**2 + eps**2)**0.5)


def degraded_scalar(f, f_prev, g):
    fplus = heaviside(f_prev)*f
    fminus = f - fplus
    return g*fplus + fminus


def degraded_stress(ε,ε_prev,g,ν):
    λoverμ = 2*ν/(1-2*ν)
    D = ufl.shape(ε)[0]
    I = ufl.Identity(D)
    i,j,k,l = ufl.indices(4)
    σ = λoverμ*ufl.tr(ε)*I + 2*ε
    P = projection_tensor(ε_prev)
    σplus = λoverμ*heaviside(ufl.tr(ε_prev))*ufl.tr(ε)*I + \
        2*ufl.as_tensor(P[i,j,k,l]*ε[k,l],(i,j))
    σminus = σ - σplus
    return g*σplus + σminus


def degraded_stress_jakub(u, u_prev, g, ν):
    λoverμ = 2*ν/(1-2*ν); I = ufl.Identity(len(u))
    i,j,k,l = ufl.indices(4)
    σ = (λoverμ+2/3)*ufl.tr(ε(u))*I + 2*εD(u)
    P = projection_tensor(εD(u_prev))
    σplus = (λoverμ+2/3)*heaviside(ufl.tr(ε(u_prev)))*ufl.tr(ε(u))*I + \
        2*ufl.as_tensor(P[i,j,k,l]*εD(u)[k,l],(i,j))
    σminus = σ - σplus
    return g*σplus + σminus


def degraded_deviatoric(εD,ε_prev, g, ν):
    P = projection_tensor(ε_prev)
    i,j,k,l = ufl.indices(4)
    σD = 2*εD
    σDplus = 2*ufl.as_tensor(P[i,j,k,l]*εD[k,l],(i,j))
    σDminus = σD - σDplus
    return g*σDplus + σDminus
    


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



def general_projection_tensor(σ0, σplus):
    D = ufl.shape(σ0)[0]
    invσ0 = ufl.inv(σ0)
    i,j,k,l = ufl.indices(4)
    P = ufl.as_tensor(σplus[i,j]*invσ0[l,k],(i,j,k,l))/D
    return P