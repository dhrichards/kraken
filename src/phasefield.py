import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
from invariants import matrix_function, eigenstate
from common import ε, stress, free_energy




def positive_part(x):
    return ufl.max_value(x,0)

def negative_part(x):
    return ufl.min_value(x,0)


def degradation(d,k=1e-5):
    return (1-d)**2 + k


def γ(d,l):
    return 0.5/l * (d**2 + l**2 * ufl.inner(ufl.grad(d), ufl.grad(d)))


def free_energy_plus(u,ν):
# based on alternative formulation, equivalent to below
    λoverμ = 2*ν/(1-2*ν)
    εplus = matrix_function(ε(u),positive_part)
    return 0.5*λoverμ*positive_part(ufl.tr(ε(u)))**2 + \
            ufl.inner(εplus,εplus)


def degraded_free_energy(u,d,ν,ψcritstar):
    # ψplus = positive_part(free_energy_plus(u,ν)-ψcritstar)
    ψplus = free_energy_plus(u,ν)
    ψminus = free_energy(u,ν) - ψplus
    # return degradation(d)*(ψplus) + (ψminus)
    return degradation(d)*(ψplus-ψcritstar) + (ψminus+ψcritstar)

# def free_energy_plus(u,ν):
# ## based on formulation in Miehle "Thermodynamically consistent phase-field models of fracture"
#     λoverμ = 2*ν/(1-2*ν)
#     εi, eigvecs = eigenstate(stress(u,ν))

#     return 0.5*λoverμ*positive_part(sum(εi))**2 + \
#             positive_part(εi[0])**2 + positive_part(εi[1])**2



def degraded_stress(u,d,ν):
    λoverμ = 2*ν/(1-2*ν); I = ufl.Identity(len(u))
    σ = λoverμ*ufl.tr(ε(u))*I + 2*ε(u)   
    σplus = λoverμ*positive_part(ufl.tr(ε(u)))*I + \
        2*matrix_function(ε(u),positive_part)
    σminus = λoverμ*negative_part(ufl.tr(ε(u)))*I + \
        2*matrix_function(ε(u),negative_part)

    return degradation(d)*σplus + σminus

def degraded_pressure(p,d):
    pplus = positive_part(p)
    pminus = p - pplus
    return degradation(d)*pplus + pminus


def history_function(ψ,ψcrit,Hprev):
    return ufl.max_value(positive_part(ψ-ψcrit),Hprev)


def initilise_history_function(msh):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    H = fem.Function(V)

    H.interpolate(0.0)
    return H

     


def solve(msh,bc_func,uh,material,H=0.0):
    V = fem.functionspace(msh, ("Lagrange", 1))

    bcs = bc_func(V)

    d = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    ψplus = free_energy_plus(uh,material.ν)
    H = history_function(ψplus,material.ψcritstar,H)
    # H = free_energy(uh,material.ν)

    C3 = material.C3; l = material.l

    a = ((1+2*C3*l*H)*d*v + l**2*ufl.inner(ufl.grad(d), ufl.grad(v))) * ufl.dx
    L = 2*C3*l*H*v * ufl.dx 

    problem = LinearProblem(a, L, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dh = problem.solve()

    dh.name = "d"

    return dh,H


