import ufl
from math import sqrt
from .invariants import eigenstate, matrix_function
from .maths_functions import positive_part, tr_e

def λoverμ(ν):
    return 2*ν/(1-2*ν)

def Koverμ(ν):
    return λoverμ(ν) + 2/3

def Koverλ(ν):
    return (1+ν)/(3*ν)

def Eoverμ(ν):
    return 2*(1+ν)

def cauchy_stress(ε,ν):
    D = ufl.shape(ε)[0]
    return λoverμ(ν)*tr_e(ε)*ufl.Identity(D) + 2*ε 


def free_energy(ε,ν):
    εD = ufl.dev(ε)
    return 0.5*Koverμ(ν)*tr_e(ε)**2 + ufl.inner(εD,εD) 



# def stress_plus_spectral(ε,ν):
#     D = ufl.shape(ε)[0]
#     I = ufl.Identity(D)
#     εplus = matrix_function(ε,positive_part)
#     return λoverμ(ν)*positive_part(tr_e(ε))*I + 2*εplus

# def free_energy_plus_spectral(ε,ν):
#     ##Spectral:
#     εplus = matrix_function(ε,positive_part)
#     return 0.5*λoverμ(ν)*positive_part(tr_e(ε))**2 \
#             + ufl.inner(εplus,εplus) 




def free_energy_plus_dp(ε, ν, γ=sqrt(3), eps=1e-14):
    K = Koverμ(ν)
    εD2 = ufl.inner(ufl.dev(ε), ufl.dev(ε)) + eps
    
    
    
    ψ1 = 0.5*K*tr_e(ε)**2 + εD2
    ψ2 = (K*γ*tr_e(ε) + 2*ufl.sqrt(εD2))**2 / (2*(K*γ**2 + 2))

    return ufl.conditional(ufl.lt(ufl.sqrt(εD2), tr_e(ε)/γ), ψ1,
                        ufl.conditional(ufl.lt(ufl.sqrt(εD2), -γ*K/2*tr_e(ε)), 0.0, 
                                         ψ2))
  




def stress_plus_dp(ε, ν, γ=sqrt(3), eps=1e-14):
    K = Koverμ(ν)
    εD2 = ufl.inner(ufl.dev(ε), ufl.dev(ε)) + eps
    D = ufl.shape(ε)[0]
    δ = ufl.Identity(D)

    σ1 = K*tr_e(ε)*δ + 2*ufl.dev(ε)
    σ2 = (K*γ*δ + ufl.dev(ε)/ufl.sqrt(εD2))/(K*γ**2 + 2)

    return ufl.conditional(ufl.lt(ufl.sqrt(εD2), tr_e(ε)/γ), σ1,
                        ufl.conditional(ufl.lt(ufl.sqrt(εD2), -γ*K/2*tr_e(ε)), 0.0*δ, 
                                         σ2))






def free_energy_plus_lo(ε,ν):

    λ,M = eigenstate(ε)
    λ_mid = (ε[0,0] + ε[1,1])/2
    λ = [λ[0], λ_mid, λ[1]]

    psi1 = free_energy(ε,ν)
    psi2 = free_energy(ε,ν) - Eoverμ(ν)*λ[0]**2/2
    psi3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    return ufl.conditional(ufl.gt(λ[0],0),psi1,
            ufl.conditional(ufl.gt(λ[1] + ν*λ[0],0),psi2,
             ufl.conditional(ufl.gt((1-ν)*λ[2] + ν*(λ[0]+λ[1]),0),psi3,
                             0)))

def free_energy_plus_lo_3d(ε,ν):
    Eoverμ = 2*(1+ν)
    λ,M = eigenstate(ε)

    psi1 = free_energy(ε,ν)
    psi2 = free_energy(ε,ν) - Eoverμ*λ[0]**2/2
    psi3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    return ufl.conditional(ufl.gt(λ[0],0),psi1,
            ufl.conditional(ufl.gt(λ[1] + ν*λ[0],0),psi2,
             ufl.conditional(ufl.gt((1-ν)*λ[2] + ν*(λ[0]+λ[1]),0),psi3,
                             0)))


def stress_plus_lo(ε,ν):
    E = Eoverμ(ν)
    λ,M = eigenstate(ε)
    λ_mid = (ε[0,0] + ε[1,1])/2
    λ = [λ[0], λ_mid, λ[1]]
    M = [M[0], 0, M[1]]

    #M_mid is 0 in 2D

    stress1 = cauchy_stress(ε,ν)
    stress2 = cauchy_stress(ε,ν) - E*λ[0]*M[0]

    # psi3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    stress3 = 2*(1+ν)/((1-2*ν)*(1-ν**2))*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])*(
        (1-ν)*M[2] + ν*M[0])

    return ufl.conditional(ufl.gt(λ[0],0),stress1,
            ufl.conditional(ufl.gt(λ[1] + ν*λ[0],0),stress2,
             ufl.conditional(ufl.gt((1-ν)*λ[2] + ν*(λ[0]+λ[1]),0),stress3,
                             ufl.zero(ufl.shape(ε)))))


def stress_plus_lo_3d(ε,ν):
    E = 2*(1+ν)

    λ,M = eigenstate(ε)
    

    stress1 = cauchy_stress(ε,ν)
    stress2 = cauchy_stress(ε,ν) - E*λ[0]*M[0]

    # psi3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    stress3 = 2*(1+ν)/((1-2*ν)*(1-ν**2))*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])*(
        (1-ν)*M[2] + ν*M[1] + ν*M[0])

    
    return ufl.conditional(ufl.gt(λ[0],0),stress1,
            ufl.conditional(ufl.gt(λ[1] + ν*λ[0],0),stress2,
             ufl.conditional(ufl.gt((1-ν)*λ[2] + ν*(λ[0]+λ[1]),0),stress3,
                                ufl.zero(ufl.shape(ε)))))






def degradation_default(d,k=1e-12):
    return (1-k)*(1-d)**2 + k


def degradation_rational(d, m=3.0):
    return ((1-d)**2)/((1-d)**2 + m*d*(1+d))


def degradation_Lo2023(d,q=1.0,k=1e-5):
    ϕ = 1-d
    g = (q+1)*(1 - (q/(q+1))**(ϕ**2) )
    return (1-k)*g + k

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw

