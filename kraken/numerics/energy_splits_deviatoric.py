import ufl
from math import sqrt
from .invariants import eigenstate, matrix_function, eigenvals_deviatoric, eigenstate3_split
from .maths_functions import dev3, largest_eigenvalue, positive_part, negative_part, tensor_2d_to_3d, deviatoric2d_to_3d
from .energy_splits import λoverμ, Koverμ, Eoverμ


def free_energy(εD,trε,ν):
    εD3 = deviatoric2d_to_3d(εD)
    return 0.5*Koverμ(ν)*trε**2 + ufl.inner(εD3,εD3)


def cauchy_stress(εD,trε,ν):
    D = ufl.shape(εD)[0]
    I = ufl.Identity(D)
    return Koverμ(ν)*trε*I + 2*εD


def free_energy_plus_dp(εD, trε, ν, B = -0.3, eps=1e-12):
    # B = -0.4
    K = Koverμ(ν)
    I1 = trε
    εD3 = deviatoric2d_to_3d(εD)
    J2 = 0.5*ufl.inner(εD3, εD3) + eps
    
    ψ1 = 0.5*K*I1**2 + 2*J2
    ψ2 = (-3*B*K*I1 + 2*ufl.sqrt(J2))**2 / (18*B**2*K + 2)

    ψ = ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2), I1), ψ1,
                        ufl.conditional(2*ufl.sqrt(J2) < 3*B*K*I1, 0.0, 
                                         ψ2))
    return ψ



def stress_plus_dp(εD, trε, ν, B = -0.3, eps=1e-12):
    #-0.3 is good
    K = Koverμ(ν)
    I1 = trε
    εD3 = deviatoric2d_to_3d(εD)
    J2 = 0.5*ufl.inner(εD3, εD3) + eps
    δ = ufl.Identity(ufl.shape(εD)[0])
    

    σ1 = K*I1*δ + 2*εD
    σ2 = ((18*B**2*K**2*I1 - 12*B*K*ufl.sqrt(J2))*δ \
          +(4 - 6*B*K*I1/ufl.sqrt(J2))*εD
          )/(18*B**2*K + 2)

    return ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2), I1), σ1,
                        ufl.conditional(2*ufl.sqrt(J2) < 3*B*K*I1, 0.0*δ, 
                                         σ2))

def deviatoric_stress_plus_dp(εD, trε, ν, B = -0.3, eps=1e-12):
    #-0.3 is good
    K = Koverμ(ν)
    I1 = trε
    εD3 = deviatoric2d_to_3d(εD)
    J2 = 0.5*ufl.inner(εD3, εD3) + eps
    δ = ufl.Identity(ufl.shape(εD)[0])
    

    σD1 = 2*εD
    σD2 = ((4 - 6*B*K*I1/ufl.sqrt(J2))*εD
          )/(18*B**2*K + 2)

    return ufl.conditional(ufl.lt(-6*B*ufl.sqrt(J2), I1), σD1,
                        ufl.conditional(2*ufl.sqrt(J2) < 3*B*K*I1, 0.0*δ, 
                                         σD2))



# def free_energy_plus_lo(εD,trε,ν, eps = 1e-16):

#     I1 = trε
#     J2 = 0.5*ufl.inner(εD, εD) + eps

#     E = Eoverμ(ν)
#     #λ2>λ1>λ0

#     ϵ1 = I1/3 - ufl.sqrt(J2)
#     ϵ2 = I1/3
#     ϵ3 = I1/3 + ufl.sqrt(J2)

#     cond1 = ufl.gt(ϵ1,0)
#     cond2 = ufl.gt(ϵ2 + ν*ϵ1,0)
#     cond3 = ufl.gt((1-ν)*ϵ3 + ν*(ϵ1 + ϵ2),0)

#     # ψ1 = E*ν/(2*(1+ν)*(1-2*ν))*(ϵ1 + ϵ2 + ϵ3)**2 \
#     #     + E/(2*(1+ν))*(ϵ1**2 + ϵ2**2 + ϵ3**2)

#     # ψ2 = E*ν/(2*(1+ν)*(1-2*ν))*(ϵ3 + ϵ2 + 2*ν*ϵ1)**2 \
#     #     + E/(2*(1+ν))*( (ϵ3 + ν*ϵ1)**2 + (ϵ2 + ν*ϵ1)**2)

#     # ψ3 = E/(2*(1-ν**2)*(1-2*ν))*((1-ν)*ϵ3 + ν*ϵ1 + ν*ϵ2)**2 
#     λ = [ϵ1, ϵ2, ϵ3]

#     ψ1 = free_energy(εD, trε, ν)
#     ψ2 = free_energy(εD, trε, ν) - Eoverμ(ν)*λ[0]**2/2
#     ψ3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    

#     # psi1 = free_energy(εD,trε,ν)
#     # psi2 = free_energy(εD,trε,ν) - Eoverμ(ν)*λ[0]**2/2
#     # psi3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

#     return ufl.conditional(cond1,ψ1,
#             ufl.conditional(cond2,ψ2,
#              ufl.conditional(cond3,ψ3,
#                              0)))


def free_energy_plus_lo(εD,trε,ν):

    Eoverμ = 2*(1+ν)
    λ,M = eigenstate3_split(trε, εD)

    psi1 = free_energy(εD, trε, ν)
    psi2 = free_energy(εD, trε, ν) - Eoverμ*λ[0]**2/2
    psi3 = (1+ν)*((1-ν)*λ[2]+ν*λ[1] +ν*λ[0])**2/((1-2*ν)*(1-ν**2))

    return ufl.conditional(ufl.gt(λ[0],0),psi1,
            ufl.conditional(ufl.gt(λ[1] + ν*λ[0],0),psi2,
             ufl.conditional(ufl.gt((1-ν)*λ[2] + ν*(λ[0]+λ[1]),0),psi3,
                             0)))


def stress_plus_lo(εD,trε,ν, eps = 1e-16):
    I1 = trε
    J2 = 0.5*ufl.inner(εD, εD) + eps
    I = ufl.Identity(ufl.shape(εD)[0])

    E = Eoverμ(ν)
    
    cond1 = ufl.gt(I1/3 - ufl.sqrt(J2),0)
    cond2 = ufl.gt(I1/3 + ν*(I1/3 - ufl.sqrt(J2)),0)
    cond3 = ufl.gt(ν*(2*I1/3 - ufl.sqrt(J2)) + (1-ν)*(I1/3 + ufl.sqrt(J2)),0)

    σ1 = cauchy_stress(εD,trε,ν)
    σ2 = (-2*I*ufl.sqrt(J2)*(ν + 1)*(2*I1 + 3*ufl.sqrt(J2) + 2*ν*(I1 - 3*ufl.sqrt(J2))) + 3*εD*(2*ν - 1)*(I1*ν + I1 + 3*ufl.sqrt(J2)*(1 - ν)))/(9*ufl.sqrt(J2)*(2*ν - 1))
    σ3 = ν*(-2*I*ufl.sqrt(J2)*(ν + 1)*(I1*ν + ν*(I1 - 3*ufl.sqrt(J2)) - (I1 + 3*ufl.sqrt(J2))*(ν - 1)) + 3*εD*(2*ν - 1)*(I1*ν + I1 - 6*ufl.sqrt(J2)*ν + 3*ufl.sqrt(J2)))/(9*ufl.sqrt(J2)*(2*ν - 1))

    return ufl.conditional(cond1,σ1,
            ufl.conditional(cond2,σ2,
             ufl.conditional(cond3,σ3,
                             0*I)))


def deviatoric_stress_plus_lo(εD,trε,ν, eps = 1e-16):
    I1 = trε
    J2 = 0.5*ufl.inner(εD, εD) + eps
    I = ufl.Identity(ufl.shape(εD)[0])

    E = Eoverμ(ν)
    
    cond1 = ufl.gt(I1/3 - ufl.sqrt(J2),0)
    cond2 = ufl.gt(I1/3 + ν*(I1/3 - ufl.sqrt(J2)),0)
    cond3 = ufl.gt(ν*(2*I1/3 - ufl.sqrt(J2)) + (1-ν)*(I1/3 + ufl.sqrt(J2)),0)

    σD1 = 2*εD
    σD2 = εD*(I1*ν + I1 + 3*ufl.sqrt(J2)*(1 - ν))/(3*ufl.sqrt(J2))
    σD3 = εD*ν*(I1*ν + I1 - 6*ufl.sqrt(J2)*ν + 3*ufl.sqrt(J2))/(3*ufl.sqrt(J2))

    return ufl.conditional(cond1,σD1,
            ufl.conditional(cond2,σD2,
             ufl.conditional(cond3,σD3,
                             0*I)))




def free_energy_plus_spectral(εD,trε,ν):
    εD = deviatoric2d_to_3d(εD)
    ε = εD + (1/3)*trε*ufl.Identity(3)
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ(ν)*positive_part(trε)**2 \
            + ufl.inner(εplus,εplus) 




def stress_plus_spectral(εD,trε,ν):
    ε = εD + (1/3)*trε*ufl.Identity(ufl.shape(εD)[0]) # can be clipped 2d version but will give right eigenvectors
    I = ufl.Identity(ufl.shape(ε)[0])
    εplus = matrix_function(ε,positive_part)
    return λoverμ(ν)*positive_part(trε)*I + 2*εplus


def deviatoric_stress_plus_spectral(εD,trε,ν):
    ε = εD + (1/3)*trε*ufl.Identity(ufl.shape(εD)[0]) # can be clipped 2d version but will give right eigenvectors
    I = ufl.Identity(ufl.shape(ε)[0])
    λ12,_ = eigenstate(ε)
    εDzz = -εD[0,0] - εD[1,1]
    λ3 = εDzz + (1/3)*trε
    εplus = matrix_function(ε,positive_part)
    trεplus = positive_part(λ12[0]) + positive_part(λ12[1]) + positive_part(λ3)
    return 2*(εplus - (1/3)*trεplus*I)
    