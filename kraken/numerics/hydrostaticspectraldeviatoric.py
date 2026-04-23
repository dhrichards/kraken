import ufl
from math import sqrt
from .energy_splits import Koverμ, λoverμ, cauchy_stress, positive_part, negative_part
from .invariants import eigenstate, matrix_function


def free_energy_plus(ε,ν,θ,c):

    f = Mohr_criterion(ε,ν,θ,c)
    ϕ = ufl.conditional(ufl.gt(f,0),1,0)

    ψdev = free_energy_dev(ε)
    ψdev_minus = free_energy_dev_minus(ε)
    ψdev_plus = ψdev - ψdev_minus

    return 0.5*Koverμ(ν)*1.5*positive_part(ufl.tr(ε))**2 + \
            + ψdev_plus + ϕ*ψdev_minus 


def stress_plus(ε,ν,θ,c):
    f = Mohr_criterion(ε,ν,θ,c)
    ϕ = ufl.conditional(ufl.gt(f,0),1,0)

    σdev = 2*ufl.dev(ε)
    σdev_minus = stress_dev_minus(ε)
    σdev_plus = σdev - σdev_minus

    return Koverμ(ν)*1.5*positive_part(ufl.tr(ε))*ufl.Identity(2) + \
            + σdev_plus + ϕ*σdev_minus

    

def free_energy_dev_minus(ε):

    λ,M = eigenstate(ε)
    λ_mid = (ε[0,0] + ε[1,1])/2
    λ = [λ[0], λ_mid, λ[1]]

    return (2/3)*(negative_part(λ[0])**2 + negative_part(λ[1])**2 + negative_part(λ[2])**2)


def free_energy_dev(ε):

    λ,M = eigenstate(ε)
    λ_mid = (ε[0,0] + ε[1,1])/2
    λ = [λ[0], λ_mid, λ[1]]

    return (2/3)*(λ[0]**2 + λ[1]**2 + λ[0]**2) \
            - (2/3)*(λ[0]*λ[1] + λ[1]*λ[2] + λ[0]*λ[2])




def Mohr_criterion(ε,ν,θ,c):
    λ,M = eigenstate(ε)
    λ_mid = (ε[0,0] + ε[1,1])/2
    λ = [λ[0], λ_mid, λ[1]]
    #\lambda_3 >= \lambda_2 >= \lambda_1

    return (λoverμ(ν)+1)*(λ[2]+λ[0])/(λ[2]-λ[0])*ufl.sin(θ) \
            - c/(λ[2]-λ[0])*ufl.cos(θ) + 1



    