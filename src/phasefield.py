import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
from invariants import matrix_function, eigenstate
import elasticity as el

def ε(u):
    return ufl.sym(ufl.grad(u))

def stress(u,ν):
    λoverμ = 2*ν/(1-2*ν)
    return λoverμ*ufl.tr(ε(u))*ufl.Identity(len(u)) + 2*ε(u)


def free_energy(ε,ν):
    λoverμ = 2*ν/(1-2*ν)
    return 0.5*λoverμ*ufl.tr(ε)**2 + ufl.inner(ε,ε)


def positive_part(x):
    return ufl.max_value(x,0)
    # return ufl.conditional(ufl.ge(x,c),x,0.0)

def negative_part(x):
    return ufl.min_value(x,0)


def degradation(d,k=1e-5):
    return (1-d)**2 + k


def γ(d,l):
    return 0.5/l * (d**2 + l**2 * ufl.inner(ufl.grad(d), ufl.grad(d)))


def free_energy_plus(ε,ν):
# based on alternative formulation, equivalent to below
    λoverμ = 2*ν/(1-2*ν)
    εplus = matrix_function(ε,positive_part)
    return 0.5*λoverμ*positive_part(ufl.tr(ε))**2 + \
            ufl.inner(εplus,εplus)


def degraded_free_energy(ε,d,ν,ψcritstar):
    ψplus = free_energy_plus(ε,ν)-ψcritstar
    # ψplus = free_energy_plus(u,ν)
    ψminus = free_energy(ε,ν) - ψplus
    return degradation(d)*(ψplus) + (ψminus)
    # return degradation(d)*(ψplus-ψcritstar) + (ψminus+ψcritstar)

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


def history_function(ε,material,Hprev):
    ψp = free_energy_plus(ε,material.ν)
    return ufl.max_value(positive_part(ψp-material.ψcritstar),Hprev)


# def initilise_history_function(msh):
#     V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

#     H = fem.Function(V)

#     H.interpolate(0.0)
#     return H

     


def solve(msh,bc_func,uh,material,Hprev=None):
    V = fem.functionspace(msh, ("Lagrange", 1))

    if Hprev is None:
        Hprev = fem.Constant(msh, default_scalar_type(0.0))

    H = history_function(ε(uh),material,Hprev)

    bcs = bc_func(V)

    d = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    C3 = material.C3; l = material.l

    # a = ((1+2*C3*l*H)*d*v + l**2*ufl.inner(ufl.grad(d), ufl.grad(v))) * ufl.dx
    # L = 2*C3*l*H*v * ufl.dx 

    F = (ufl.inner(d,v) + l**2*ufl.inner(ufl.grad(d), ufl.grad(v)) \
         - C3*l*2*(1-d)*H*v) * ufl.dx
    
    a, L = ufl.lhs(F), ufl.rhs(F)
    

    problem = LinearProblem(a, L, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dh = problem.solve()

    dh.name = "d"

    return dh



def minimisation(msh,bcfuncs,material,dh=0.0,uh=None,pw=None,Hprev=0.0,max_its=100,tol=1e-4):


    




    L2_old = 0.0
    for i in range(max_its):

        uh = el.solve(msh,bcfuncs[0],material,dh,uh,pw)
        dh = solve(msh,bcfuncs[1],uh,material,Hprev)

        L2_ = ufl.inner(dh,dh)*ufl.dx
        L2_rank = fem.assemble_scalar(fem.form(L2_))
        L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

        error_L2 = np.abs(L2 - L2_old)
        print(f"iteration {i}, error {error_L2}")
        
        if error_L2 < tol:
            break

        L2_old = L2


    return uh,dh

    

