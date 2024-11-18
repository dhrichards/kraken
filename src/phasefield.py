import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from invariants import matrix_function


def ε(u):
    return ufl.sym(ufl.grad(u))

def positive_part(x):
    return ufl.max_value(x,0)

def negative_part(x):
    return ufl.min_value(x,0)

# def positive_part_tensor(A): # Don't need this as matrix function operates on eigenvalues
#     return ufl.elem_op(positive_part, A)

# def negative_part_tensor(A):
#     return ufl.elem_op(negative_part, A)


def degradation(d):
     return (1-d)**2


def free_energy_plus(u,ν):
    λoverμ = 2*ν/(1-2*ν)
    εDplus = matrix_function(ufl.dev(ε(u)),positive_part)
    return (λoverμ+2/3)/2*positive_part(ufl.tr(ε(u)))^2 + \
            ufl.inner(εDplus,εDplus)

def degraded_stress(u,d,ν):
    λoverμ = 2*ν/(1-2*ν)
    σ = λoverμ*ufl.tr(ε(u))*ufl.Identity(len(u)) + 2*ε(u)
    p = -(λoverμ+2/3)*ufl.tr(ε(u))
    σplus = positive_part(-p)*ufl.Identity(len(u)) \
        + 2*matrix_function(ufl.dev(ε(u)),positive_part)
    σminus = σ - σplus

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

     


def phase_field(msh,uh,H,material):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    d = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    ψplus = free_energy_plus(uh,material.ν)
    H = history_function(ψplus,material.ψcrit,H)

    C3 = material.C3; l = material.l

    a = ((1-2*C3*l)*H*d*v + l**2*ufl.inner(ufl.grad(d), ufl.grad(v))) * ufl.dx
    L = 2*C3*l*H*v * ufl.dx 

    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dh = problem.solve()

    return dh,H
        


