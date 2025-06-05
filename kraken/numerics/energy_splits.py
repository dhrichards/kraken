import ufl
from .invariants import eigenstate, matrix_function

def λoverμ(ν):
    return 2*ν/(1-2*ν)

def cauchy_stress(ε,ν):
    D = ufl.shape(ε)[0]
    return λoverμ(ν)*ufl.tr(ε)*ufl.Identity(D) + 2*ε 


def stress_plus_spectral(ε,ν):
    I = ufl.Identity(ufl.shape(ε)[0])
    εplus = matrix_function(ε,positive_part)
    return λoverμ(ν)*positive_part(ufl.tr(ε))*I + 2*εplus



def free_energy(ε,ν):
    return 0.5*λoverμ(ν)*ufl.tr(ε)**2 + ufl.inner(ε,ε) 

def positive_part(x,eps=1e-8):
    # return 0.5*(x + (x**2 + eps**2)**0.5)
    return ufl.max_value(0.0,x)
    # return ufl.conditional(ufl.gt(x,0),x,0)
    # return 0.5*(x + abs(x))
    # return 0.5*(x + ufl.sign(x)*x)


def negative_part(x,eps=1e-6):
    return 0.5*(x-abs(x))
    # return 0.5*(x - (x**2 + eps**2)**0.5)




def free_energy_plus_amor(ε,ν):
    κ = λoverμ(ν) + 2/3
    return 0.5*κ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(ufl.dev(ε),ufl.dev(ε))

def free_energy_plus_star(ε,ν,γ=4):
    κ = λoverμ(ν) + 2/3
    return 0.5*κ*(positive_part(ufl.tr(ε))**2 \
                  - γ*negative_part(ufl.tr(ε))**2) \
            + ufl.inner(ufl.dev(ε),ufl.dev(ε)) 


def free_energy_plus_spectral(ε,ν):
    ##Spectral:
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ(ν)*positive_part(ufl.tr(ε))**2 \
            + ufl.inner(εplus,εplus) 

def free_energy_plus_stocek(ε,ν):
    κ = λoverμ(ν) + 2/3
    εplus = matrix_function(ufl.dev(ε),positive_part)
    return 0.5*κ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)


def free_energy_plus_notension(ε,ν):

    A,M = eigenstate(ε)


    α2 = positive_part(A[1])

    
    #ufl conditions to do
    # if A[0] > 0:
    #     α1 = A[0]
    # elif A[0] > (1-ν)*A[0] + ν*A[1]:
    #     α1 = A[0] + ν*A[1]/(1-ν)
    # else:
    #     α1 = 0


    α1 = ufl.conditional(ufl.gt(A[0],0),A[0],
                    ufl.conditional(ufl.gt((1-ν)*A[1] + ν*A[0],0),
                                    A[0] + ν*A[1]/(1-ν),0))
    
    α = [α1,α2]
    # Reconstruct the matrix using the modified eigenvalues
    Ece = ufl.zero(ufl.shape(ε))
    # apply UFL function on eigenvalue and synthesise matrix function
    # for M_ in M:
    for α_, M_ in zip(α,M):
        Ece += α_ * M_

    return free_energy(Ece,ν)


def free_energy_plus_lo(ε,ν):
    κ = λoverμ(ν) + 2/3

    λ,M = eigenstate(ε)


    val = (1+ν)*((1-ν)*λ[1]+ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    return ufl.conditional(ufl.gt(λ[0],0),0.5*κ*(λ[1]+λ[0])**2 + λ[1]**2 + λ[0]**2,
                           ufl.conditional(ufl.And(ufl.gt(λ[1],0),ufl.gt((1-ν)*λ[1] + ν*λ[0],0)),
                                           val,0))
                                                   

    




def free_energy_plus_basic(ε,ν):
    return free_energy(ε,ν)
