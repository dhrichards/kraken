import ufl
from math import sqrt
from .invariants import eigenstate, matrix_function
from .maths_functions import dev3, largest_eigenvalue, positive_part, negative_part, tensor_2d_to_3d

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
    return 1.5*λoverμ(ν)*ufl.tr(ε)*ufl.Identity(D) + 2*ε 

def principal_stress(ε,λ,μ):
    return largest_eigenvalue(cauchy_stress(ε,λ,μ))

def stress_plus_spectral(ε,ν):
    I = ufl.Identity(ufl.shape(ε)[0])
    εplus = matrix_function(ε,positive_part)
    return 1.5*λoverμ(ν)*positive_part(ufl.tr(ε))*I + 2*εplus



def stress_plus_amor(ε,ν):
    D = ufl.shape(ε)[0]
    I = ufl.Identity(D)
    κ = Koverμ(ν)
    return 1.5*κ*positive_part(ufl.tr(ε))*I + 2*ufl.dev(ε)



def free_energy(ε,ν):
    return 0.5*1.5*λoverμ(ν)*ufl.tr(ε)**2 + ufl.inner(ε,ε) 



def free_energy_plus_dp(ε, ν, B = -0.4, eps=1e-12):
    # B = -0.4
    K = Koverμ(ν)
    I1 = 1.5*ufl.tr(ε)
    εD = ufl.dev(ε)
    J2 = 0.5*ufl.inner(εD, εD) + eps
    
    ψ1 = 0.5*K*I1**2 + 2*J2
    ψ2 = (-3*B*K*I1 + 2*ufl.sqrt(J2))**2 / (18*B**2*K + 2)

    ψ = ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2), I1), ψ1,
                        ufl.conditional(2*ufl.sqrt(J2) < 3*B*K*I1, 0.0, 
                                         ψ2))
    return ψ





def stress_plus_dp(ε, ν, B = -0.4, eps=1e-12):
    #-0.3 is good
    # B = -0.4
    εD = ufl.dev(ε)
    K = Koverμ(ν)
    I1 = 1.5*ufl.tr(ε)
    J2 = 0.5*ufl.inner(εD, εD)
    δ = ufl.Identity(ufl.shape(ε)[0])
    

    σ1 = K*I1*δ + 2*εD
    σ2 = ((18*B**2*K**2*I1 - 12*B*K*ufl.sqrt(J2+eps))*δ \
          +(4 - 6*B*K*I1/ufl.sqrt(J2+eps))*εD
          )/(18*B**2*K + 2)

    return ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2+eps), I1), σ1,
                        ufl.conditional(2*ufl.sqrt(J2+eps) < 3*B*K*I1, 0.0*δ, 
                                         σ2))


def free_energy_plus_amor(ε,ν):
    κ = λoverμ(ν) + 2/3
    return 0.5*κ*1.5*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(ufl.dev(ε),ufl.dev(ε))

def free_energy_plus_star(ε,ν,γ=1):
    κ = Koverμ(ν)
    return 0.5*κ*1.5*(positive_part(ufl.tr(ε))**2 \
                  - γ*1.5*negative_part(ufl.tr(ε))**2) \
            + ufl.inner(ufl.dev(ε),ufl.dev(ε)) 

def stress_plus_star(ε,ν,γ=1):
    D = ufl.shape(ε)[0]
    I = ufl.Identity(D)
    κ = Koverμ(ν)
    return 1.5*(κ*positive_part(ufl.tr(ε)) - 2*γ*negative_part(ufl.tr(ε)))*I + 2*ufl.dev(ε)


def free_energy_plus_spectral(ε,ν):
    ##Spectral:
    εplus = matrix_function(ε,positive_part)
    return 0.5*1.5*λoverμ(ν)*positive_part(ufl.tr(ε))**2 \
            + ufl.inner(εplus,εplus) 




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






def clayton_driving_function(σ, σ_crit,pw=0.0):
    λ,_ = eigenstate(σ)
    Dd = 0.0
    for σa in λ:
        Dd += (positive_part(σa+pw)/σ_crit)**2 - 1.0

    return positive_part(Dd)




def degradation_default(d,k=1e-5):
    return (1-k)*(1-d)**2 + k


def degradation_rational(d, m=3.0):
    return ((1-d)**2)/((1-d)**2 + m*d*(1+d))


def degradation_Lo2023(d,q=1.0,k=1e-5):
    ϕ = 1-d
    g = (q+1)*(1 - (q/(q+1))**(ϕ**2) )
    return (1-k)*g + k

def crack_density_function(d,l,w=lambda d: d**2,cw=2):
    return  (w(d)/l + l * ufl.inner(ufl.grad(d), ufl.grad(d)))/cw



def history_function(ψplus,Hprev,ψcrit):
    return ufl.max_value(ψplus - ψcrit,Hprev)