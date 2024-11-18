import math

import dolfinx.nls.petsc
import numpy as np

import ufl
from basix.ufl import mixed_element, element
from dolfinx.cpp.la.petsc import scatter_local_vectors
from dolfinx.fem import (Function, functionspace,
                         bcs_by_block, dirichletbc, extract_function_spaces,
                         form, locate_dofs_topological, Constant)
from dolfinx.fem.petsc import (apply_lifting, apply_lifting_nest,
                               assemble_matrix, assemble_matrix_block,
                               assemble_matrix_nest, assemble_vector,
                               assemble_vector_block, assemble_vector_nest,
                               create_matrix, create_matrix_block,
                               create_matrix_nest, create_vector,
                               create_vector_block, create_vector_nest, set_bc,
                               set_bc_nest)
from dolfinx.mesh import (GhostMode, create_unit_cube, create_unit_square, create_rectangle,
                          locate_entities_boundary, CellType, cell_dim)
from dolfinx import default_real_type
from ufl import derivative, dx, inner

from mpi4py import MPI
from petsc4py import PETSc
from phasefield import degradation, ε, degraded_pressure
from elasticity import water_pressure
from nonlinear import NonlinearPDE_SNESProblem
from material import MaterialProperties

def viscosity(u, n, eps=1.e-8, A=1.0): 
    return A**(-1/n) * (ufl.inner(ε(u), ε(u)) / 2 + eps)**((1 - n) / (2 * n))



def stokes(mesh, material=MaterialProperties(), d=0.0, u=None, p=None):
    gdim = mesh.geometry.dim

    C1 = material.C1; C2 = material.C2
    ρw = material.ρw; ρi = material.ρi

    P2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
    P1_el = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)

    P2 = functionspace(mesh, P2_el)
    P1 = functionspace(mesh, P1_el)

    if u is None:
        u = Function(P2)
    if p is None:
        p = Function(P1)
    du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    def η(u):
        return viscosity(u, material.n, 1.e-8)
    
    def hat(p):
        return degraded_pressure(p, d)
    

    f = Constant(mesh, (PETSc.ScalarType(0.0), PETSc.ScalarType(-ρi/ρw)))

    # Outward-pointing unit normal to the boundary  
    n = ufl.FacetNormal(mesh)           

    # Surface measure
    ds = ufl.Measure("ds", domain=mesh)

    # Water pressure
    pw = water_pressure(mesh)

    # Phase field changes
    g = degradation(d)

    
    F = [(1/C2)*g*η(u)*inner(ε(u), ε(v)) * dx + inner(hat(-p), ufl.div(v)) * dx\
          - C1*g* (inner(f, v) + pw*ufl.div(v) )* dx + C1*g*pw*inner(n, v) * ds,
        inner(ufl.div(u), q) * dx ]
    

    J = [[derivative(F[0], u, du), derivative(F[0], p, dp)],
            [derivative(F[1], u, du), derivative(F[1], p, dp)]]
    
    P = [[J[0][0], None],
            [None, (2 * η(u))**-1 * dp * q * ufl.dx]]
    
    F, J, P = form(F), form(J), form(P)


    Jmat = create_matrix_nest(J)
    Pmat = create_matrix_nest(P)
    Fvec = create_vector_nest(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-15, max_it=10)
    nested_IS = Jmat.getNestISs()
    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], [], P=P)
    snes.setFunction(problem.F_nest, Fvec)
    snes.setJacobian(problem.J_nest, J=Jmat, P=Pmat)

    x = create_vector_nest(F)

    assert x.getType() == "nest"
    for x_soln_pair in zip(x.getNestSubVecs(), (u, p)):
        x_sub, soln_sub = x_soln_pair
        soln_sub.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        soln_sub.x.petsc_vec.copy(result=x_sub)
        x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Solve nonlinear problem
    snes.solve(None, x)
    assert snes.getKSP().getConvergedReason() > 0

    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p



    



