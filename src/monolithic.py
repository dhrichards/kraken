import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type, la, default_real_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.log import LogLevel, set_log_level
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
import phasefield as pf
from common import *
import basix.ufl as bufl
import nonlinear



def solve(msh, bc_func, material, pw=None):
    # V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    vel = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type)
    sel = bufl.element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    # mixed_el = bufl.mixed_element([vel, sel])
    # V = fem.functionspace(msh, mixed_el)
    V_u = fem.functionspace(msh, vel)
    V_d = fem.functionspace(msh, sel)


    bcs = bc_func(V_u)

    # Pull properties out
    ρratio = material.ρratio; C1 = material.C1; ν = material.ν
    C3 = material.C3; l = material.l

    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)
    # pw = water_pressure(msh)

    if pw is None:
        pw = lambda u: water_pressure(msh,u)

    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))



    u = fem.Function(V_u, name="Displacement")
    d = fem.Function(V_d, name="Damage")

    v = ufl.TestFunction(V_u)
    e = ufl.TestFunction(V_d)

    du = ufl.TrialFunction(V_u)
    de = ufl.TrialFunction(V_d)

    # need upper/lower bound for the damage field
    d_lb = fem.Function(V_d, name="Lower bound")
    d_ub = fem.Function(V_d, name="Upper bound")
    d_ub.x.array[:] = 1
    d_lb.x.array[:] = 0


    internal_energy = (pf.degraded_free_energy(u,d,ν,material.ψcritstar)\
                        + (1/C3)*pf.γ(d,l)) * ufl.dx
    
    F = [ufl.derivative(internal_energy, u, v),
         ufl.derivative(internal_energy, d, e)]
    
    J = [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], d, de)],
         [ufl.derivative(F[1], u, du), ufl.derivative(F[1], d, de)]]
    
    P = [[J[0][0], None],
            [None, None]]
    
    u,d = nonlinear.nested_solve(F, J, u, d, bcs)
    


    return u,d
