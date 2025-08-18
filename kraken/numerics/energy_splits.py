import ufl
from math import sqrt
from .invariants import eigenstate, matrix_function
from .maths_functions import dev3, largest_eigenvalue, positive_part, negative_part

def λoverμ(ν):
    return 2*ν/(1-2*ν)

def Koverμ(ν):
    return λoverμ(ν) + 2/3

def cauchy_stress(ε,ν):
    D = ufl.shape(ε)[0]
    return λoverμ(ν)*ufl.tr(ε)*ufl.Identity(D) + 2*ε 

def principal_stress(ε,λ,μ):
    return largest_eigenvalue(cauchy_stress(ε,λ,μ))



def stress_plus_spectral(ε,ν):
    I = ufl.Identity(ufl.shape(ε)[0])
    εplus = matrix_function(ε,positive_part)
    return λoverμ(ν)*positive_part(ufl.tr(ε))*I + 2*εplus


def stress_plus_amor(ε,ν):
    D = ufl.shape(ε)[0]
    I = ufl.Identity(D)
    κ = Koverμ(ν)
    return κ*positive_part(ufl.tr(ε))*I + 2*dev3(ε)



def free_energy(ε,ν):
    return 0.5*λoverμ(ν)*ufl.tr(ε)**2 + ufl.inner(ε,ε) 



def free_energy_plus_dplike(ε, ν, γ=1):
    εD = dev3(ε)
    normεD = ufl.sqrt(ufl.inner(εD, εD)+1e-8)
    K = Koverμ(ν)

    ψ1 = free_energy(ε, ν)
    ψ2 = (K*γ*ufl.tr(ε) + 2*normεD)**2 / (2*(K*γ**2 + 2))

    return ufl.conditional(ufl.lt(normεD, ufl.tr(ε)/γ), ψ1,
                           ufl.conditional(ufl.gt(normεD, -K*γ*ufl.tr(ε)/2), ψ2, 0)) 
                                        



def free_energy_plus_dp(ε, ν, B = -0.8/sqrt(3.0)):
    K = Koverμ(ν)
    I1 = ufl.tr(ε)
    εD = dev3(ε)
    J2 = 0.5*ufl.inner(εD, εD)
    
    ψ1 = 0.5*K*I1**2 + 2*J2
    ψ2 = (-3*B*K*I1 + 2*ufl.sqrt(J2+1e-9))**2 / (18*B**2*K + 2)

    ψ = ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2+1e-9), I1), ψ1,
                        ufl.conditional(2*ufl.sqrt(J2+1e-9) < 3*B*K*I1, 0.0, 
                                         ψ2))
    return ψ


def stress_plus_dp(ε, ν, B = -0.8/sqrt(3.0)):
    εD = dev3(ε)
    K = Koverμ(ν)
    I1 = ufl.tr(ε)
    J2 = 0.5*ufl.inner(εD, εD)
    δ = ufl.Identity(ufl.shape(ε)[0])
    

    σ1 = K*I1*δ + 2*εD
    σ2 = ((18*B**2*K**2*I1 - 12*B*K*ufl.sqrt(J2+1e-9))*δ \
          +(4 - 6*B*K*I1/ufl.sqrt(J2+1e-6))*εD
          )/(18*B**2*K + 2)

    return ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2+1e-9), I1), σ1,
                        ufl.conditional(2*ufl.sqrt(J2+1e-9) < 3*B*K*I1, 0.0*δ, 
                                         σ2))

    



def free_energy_plus_amor(ε,ν):
    κ = λoverμ(ν) + 2/3
    return 0.5*κ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(dev3(ε),dev3(ε))

def free_energy_plus_star(ε,ν,γ=4):
    κ = Koverμ(ν)
    return 0.5*κ*(positive_part(ufl.tr(ε))**2 \
                  - γ*negative_part(ufl.tr(ε))**2) \
            + ufl.inner(dev3(ε),dev3(ε)) 


def free_energy_plus_spectral(ε,ν):
    ##Spectral:
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ(ν)*positive_part(ufl.tr(ε))**2 \
            + ufl.inner(εplus,εplus) 

def free_energy_plus_stocek(ε,ν):
    κ = Koverμ(ν)
    εplus = matrix_function(dev3(ε),positive_part)
    return 0.5*κ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)


def free_energy_plus_notension(ε,ν):

    λ,M = eigenstate(ε)


    α2 = positive_part(λ[1])

    
    #ufl conditions to do
    # if A[0] > 0:
    #     α1 = A[0]
    # elif A[0] > (1-ν)*A[0] + ν*A[1]:
    #     α1 = A[0] + ν*A[1]/(1-ν)
    # else:
    #     α1 = 0


    α1 = ufl.conditional(ufl.gt(λ[0],0),λ[0],
                    ufl.conditional(ufl.gt((1-ν)*λ[1] + ν*λ[0],0),
                                    λ[0] + ν*λ[1]/(1-ν),0))
    
    α = [α1,α2]
    # Reconstruct the matrix using the modified eigenvalues
    Ece = ufl.zero(ufl.shape(ε))
    # apply UFL function on eigenvalue and synthesise matrix function
    # for M_ in M:
    for α_, M_ in zip(α,M):
        Ece += α_ * M_

    return free_energy(Ece,ν)


def free_energy_plus_lo(ε,ν):

    λ,M = eigenstate(ε)

    psi1 = free_energy(ε,ν)
    psi2 = (1+ν)*((1-ν)*λ[1]+ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    return ufl.conditional(ufl.gt(λ[0],0),psi1,
                           ufl.conditional(ufl.gt((1-ν)*λ[1] + ν*λ[0],0),
                                           psi2,0))
                                                   

def stress_plus_lo(ε,ν):
    κ = Koverμ(ν)

    λ,M = eigenstate(ε)


    # val1 = 0.5*κ*(λ[1]+λ[0])**2 + λ[1]**2 + λ[0]**2

    stress1 = cauchy_stress(ε,ν)

    stress2 = 2*(1+ν)/((1-2*ν)*(1-ν**2))*(
        (1-ν)*((1-ν)*λ[1]+ν*λ[0])*M[1] \
        + ν*((1-ν)*λ[1]+ν*λ[0])*M[0])

    return ufl.conditional(ufl.gt(λ[0],0),stress1,
                           ufl.conditional(ufl.gt((1-ν)*λ[1] + ν*λ[0],0),
                                           stress2,0*ufl.Identity(2)))




def free_energy_plus_basic(ε,ν):
    return free_energy(ε,ν)




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




def degraded_free_energy(ε,g,ν,ψcrit,free_energy_plus=free_energy_plus_spectral):
    ψplus = (free_energy_plus(ε,ν)-ψcrit)
    # # ψplus = free_energy_plus(u,ν)
    ψminus = free_energy(ε,ν) - ψplus
    return g*ψplus + ψminus



def history_function(ε,Hprev,ν,ψcrit,free_energy_plus=free_energy_plus_spectral):
    ψp = free_energy_plus(ε,ν) - ψcrit
    return ufl.max_value(ψp,Hprev)
    # return ufl.conditional(ufl.gt(ψp,Hprev),ψp,Hprev)