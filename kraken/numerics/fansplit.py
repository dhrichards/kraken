import ufl
from math import sqrt
from .energy_splits import Koverμ, λoverμ, cauchy_stress, positive_part, negative_part
from .invariants import eigenstate, matrix_function


def rotation_matrix(θ):
    return ufl.as_tensor([[ufl.cos(θ), ufl.sin(θ)],
                          [-ufl.sin(θ), ufl.cos(θ)]])


def stress_plus(ε,ν):
    λ,P = eigenstate(ε)
    print(ufl.shape(P[1]))
   
    e_p = ufl.as_tensor([[λ[0], 0],
                       [0, λ[1]]])
    
    GcI = 1.0 # normalized elswhere
    GcII = 2.0 # lets say
    
    Fs = []
    σ_pI_list = []
    σ_pII_list = []
    σ_minus_list = []
    for θ in [0, ufl.pi/4]:
        Q = rotation_matrix(θ)
        e_c = ufl.dot(ufl.dot(Q, e_p), ufl.transpose(Q))

        e_cI = tension_decomposition(e_c,ν)
        e_cII = 0.5*ufl.as_tensor([[0, e_c[0,1]],
                                      [e_c[1,0], 0]])
        e_c_minus = ε - e_cI - e_cII

        S = ufl.dot(Q,ufl.transpose(P))

        e_pI = ufl.dot(ufl.dot(ufl.transpose(S), e_cI), S)
        e_pII = ufl.dot(ufl.dot(ufl.transpose(S), e_cII), S)
        e_minus = ufl.dot(ufl.dot(ufl.transpose(S), e_c_minus), S)

        σ_pI_list.append(λoverμ(ν)*ufl.tr(e_pI)*ufl.Identity(2) + 2*e_pI)
        σ_pII_list.append(λoverμ(ν)*ufl.tr(e_pII)*ufl.Identity(2) + 2*e_pII)
        σ_minus_list.append(λoverμ(ν)*ufl.tr(e_minus)*ufl.Identity(2) + 2*e_minus)

        Fs.append(ufl.inner(σ_pI_list[-1]/GcI + σ_pII_list[-1]/GcII,ε))

    cond = ufl.gt(Fs[0], Fs[1])
    σ_pI = ufl.conditional(cond, σ_pI_list[0], σ_pI_list[1])
    σ_pII = ufl.conditional(cond, σ_pII_list[0], σ_pII_list[1])
    σ_minus = ufl.conditional(cond, σ_minus_list[0], σ_minus_list[1])
    F = ufl.conditional(cond, Fs[0], Fs[1])

    
    return σ_pI + σ_pII, F


    

def tension_decomposition(e_c,ν):
    c = λoverμ(ν)/(λoverμ(ν) + 2)
    e1 = ufl.as_tensor([[e_c[0,0], 0],
                        [0, e_c[1,1]]])
    
    e2 = ufl.as_tensor([[e_c[0,0] + c*e_c[1,1], 0],
                        [0, 0]])
    
    e3 = ufl.as_tensor([[0, 0],
                        [0, 0]])
    
    return ufl.conditional(ufl.lt(e2[0,0],0), e3,
                           ufl.conditional(ufl.gt(e_c[1,1],0), e1, e2))
    

    


