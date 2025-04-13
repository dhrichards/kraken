import ufl
from .invariants import eigenstate, matrix_function


def free_energy(ε,λ,μ):
    return 0.5*λ*ufl.tr(ε)**2 + μ*ufl.inner(ε,ε)

def positive_part(x,eps=1e-8):
    # return 0.5*(x + (x**2 + eps**2)**0.5)
    return ufl.max_value(0.0,x)
    # return ufl.conditional(ufl.gt(x,0),x,0)
    # return 0.5*(x + abs(x))
    # return 0.5*(x + ufl.sign(x)*x)


def negative_part(x,eps=1e-6):
    return 0.5*(x-abs(x))
    # return 0.5*(x - (x**2 + eps**2)**0.5)




def free_energy_plus_amor(ε,λ,μ):
    κ = λ + 2*μ/3
    return 0.5*κ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(ufl.dev(ε),ufl.dev(ε))

def free_energy_plus_star(ε,λ,μ,γ=4):
    κ = λ + 2*μ/3
    return 0.5*κ*(positive_part(ufl.tr(ε))**2 \
                  - γ*negative_part(ufl.tr(ε))**2) \
            + μ*ufl.inner(ufl.dev(ε),ufl.dev(ε)) 


def free_energy_plus_spectral(ε,λ,μ):
    ##Spectral:
    εplus = matrix_function(ε,positive_part)
    return 0.5*λ*positive_part(ufl.tr(ε))**2 + \
            μ*ufl.inner(εplus,εplus)

def free_energy_plus_stocek(ε,λ,μ):
    κ = λ + 2*μ/3
    εplus = matrix_function(ufl.dev(ε),positive_part)
    return 0.5*κ*positive_part(ufl.tr(ε))**2 + \
            μ*ufl.inner(εplus,εplus)


def free_energy_plus_notension(ε,λ,μ):

    A,M = eigenstate(ε)
    ν = λ/(2*μ + λ)


    α2 = positive_part(A[1])

    
    #ufl conditions to do
    # if A[0] > 0:
    #     α1 = A[0]
    # elif A[0] > (1-ν)*A[0] + ν*A[1]:
    #     α1 = A[0] + ν*A[1]/(1-ν)
    # else:
    #     α1 = 0


    α1 = ufl.conditional(ufl.gt(A[0],0),A[0],
                    ufl.conditional(ufl.gt((1-ν)*A[0] + ν*A[1],0),
                                    A[0] + ν*A[1]/(1-ν),0))
    
    α = [α1,α2]
    # Reconstruct the matrix using the modified eigenvalues
    Ece = ufl.zero(ufl.shape(ε))
    # apply UFL function on eigenvalue and synthesise matrix function
    # for M_ in M:
    for α_, M_ in zip(α,M):
        Ece += α_ * M_

    return free_energy(Ece,λ,μ)


def free_energy_plus_basic(ε,λ,μ):
    return free_energy(ε,λ,μ)
