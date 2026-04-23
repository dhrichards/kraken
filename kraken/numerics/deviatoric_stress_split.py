import ufl
from math import sqrt
from .energy_splits import Koverμ, λoverμ, cauchy_stress

ϕ = 16.7*ufl.pi/180  # degrees to radians
ϕ_r = 10*ufl.pi/180
E = 10e6
μ = E/(2*(1+0.325))
c = 40e3/μ
c_r = 0/μ

def stress(ε,g,ν):
    return Koverμ(ν)*ε_v(ε)*ufl.Identity(2) \
            + (g*q_hat(ε) + (1 - g)*q_r(ε,ν))*sqrt(2/3)*α_q(ε)
            


def p(ε,ν):
    return -Koverμ(ν)*ε_v(ε)

def ε_v(ε):
    return 1.5*ufl.tr(ε)

def ε_q(ε):
    e = ufl.dev(ε)
    return ufl.sqrt(2/3*ufl.inner(e,e)+1e-14)


def q(ε,g,ν):
    return g*q_hat(ε) + (1 - g)*q_r(ε,ν)

def α_q(ε):
    e = ufl.dev(ε)
    return sqrt(2/3)*e/ε_q(ε)

def q_hat(ε):
    return 3*ε_q(ε)

def q_r(ε,ν):
    return (p(ε,ν)*ufl.tan(ϕ_r) + c_r)/R_MC(ε)

def q_p(ε,ν):
    return (p(ε,ν)*ufl.tan(ϕ) + c)/R_MC(ε)

def R_MC(ε):
    θ = Lode_angle(ε)
    return 1/(3*ufl.cos(ϕ))*ufl.sin(θ + ufl.pi/3) + \
              1/3*ufl.cos(θ + ufl.pi/3)*ufl.tan(ϕ)


def Lode_angle(ε):
    s = 2*ufl.dev(ε)
    r_cubed = 9/2*ufl.tr(ufl.inner(s,s)*s) 
    q = q_hat(ε)

    cos3θ = r_cubed/(q**3 + 1e-14)
    return ufl.acos(cos3θ)


def H_t(ε,ν):
    return 1/6*(q_p(ε,ν) - q_r(ε,ν))**2

def H_slip(ε,ν):
    return 1/6*((q_hat(ε) - q_r(ε,ν))**2 - (q_p(ε,ν) - q_r(ε,ν))**2)

def ψplus(ε,ν):
    return H_t(ε,ν) + H_slip(ε,ν)