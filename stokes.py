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
from ufl import derivative, dx, inner

from mpi4py import MPI
from petsc4py import PETSc
from phasefield import degradation, ε



def stokes(msh, dh, material):


    P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    P1 = element("Lagrange", msh.basix_cell(), 1)
    
    # Create the Taylot-Hood function space
    TH = mixed_element([P2, P1])
    W = functionspace(msh, TH)

    # No slip boundary condition
    W0, _ = W.sub(0).collapse()
    # noslip = Function(W0)
    # fdim = msh.topology.dim - 1
    # facets = locate_entities_boundary(msh, fdim, boundary)
    # dofs = locate_dofs_topological((W.sub(0), W0), 1, facets)
    # bc = dirichletbc(noslip, dofs, W.sub(0))

    g = degradation(dh)

    # Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = Function(W0)
    a = form(g*(inner(ε(u), ε(v)) + inner(p, div(v)) + inner(div(u), q)) * dx)
    L = form(inner(f, v) * dx)

    # Assemble LHS matrix and RHS vector
    A = fem.petsc.assemble_matrix(a)#, bcs=bc)
    A.assemble()
    b = fem.petsc.assemble_vector(L)

    fem.petsc.apply_lifting(b, [a])#, bcs=[bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS
    fem.petsc.set_bc(b, bc)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Configure MUMPS to handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    # Compute the solution
    U = Function(W)
    try:
        ksp.solve(b, U.vector)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e

    # Split the mixed solution and collapse
    u, p = U.sub(0).collapse(), U.sub(1).collapse()

    return u, p





