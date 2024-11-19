#%%
#  Copyright (C) 2019 Nathan Sime
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

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

def nest_matrix_norm(A):
    """Return norm of a MatNest matrix"""
    assert A.getType() == "nest"
    norm = 0.0
    nrows, ncols = A.getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A.getNestSubMatrix(row, col)
            if A_sub:
                _norm = A_sub.norm()
                norm += _norm * _norm
    return math.sqrt(norm)


class NonlinearPDE_SNESProblem():
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x:
            self.soln_vars.x.array[:] = _x.array_r
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        x_array = x.getArray(readonly=True)
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x_array[offset: offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        assemble_vector_block(F, self.L, self.a, bcs=self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        assemble_matrix_block(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_block(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x:
                var_sub.x.array[:] = _x.array_r

        # Assemble
        bcs1 = bcs_by_block(extract_function_spaces(self.a, 1), self.bcs)
        for L, F_sub, a in zip(self.L, F.getNestSubVecs(), self.a):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            assemble_vector(F_sub, L)
            apply_lifting(F_sub, a, bcs=bcs1, x0=x, alpha=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = bcs_by_block(extract_function_spaces(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        assemble_matrix_nest(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_nest(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()


# mesh = create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.none),
#         #    create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.shared_facet),


mesh = create_unit_square(MPI.COMM_WORLD, 100, 101)

"""Assemble Stokes problem with Taylor-Hood elements and solve."""
gdim = mesh.geometry.dim
# P2 = VectorFunctionSpace(mesh, ("Lagrange", 2))
# P1 = FunctionSpace(mesh, ("Lagrange", 1))
from dolfinx import default_real_type

P2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
P1_el = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)

P2 = functionspace(mesh, P2_el)
P1 = functionspace(mesh, P1_el)


def boundary0(x):
    """Define boundary x = 0"""
    return np.isclose(x[0], 0.0)

def boundary1(x):
    """Define boundary x = 1"""
    return np.isclose(x[0], 1.0)



# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

# def lid_velocity_expression(x):
#     return np.row_stack((x[0]*(1-x[0]), np.zeros(x.shape[1])))


def noslip_velocity_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def initial_guess_u(x):
    u_init = np.row_stack((np.sin(x[0]) * np.sin(x[1]),
                            np.cos(x[0]) * np.cos(x[1])))
    if gdim == 3:
        u_init = np.row_stack((u_init, np.cos(x[2])))
    return u_init

def initial_guess_p(x):
    return -x[0]**2 - x[1]**3

u_bc_0 = Function(P2)
u_bc_0.interpolate(noslip_velocity_expression)

u_bc_1 = Function(P2)
u_bc_1.interpolate(lid_velocity_expression)

facetdim = mesh.topology.dim - 1
bndry_facets0 = locate_entities_boundary(mesh, facetdim, noslip_boundary)
bndry_facets1 = locate_entities_boundary(mesh, facetdim, lid)

bdofs0 = locate_dofs_topological(P2, facetdim, bndry_facets0)
bdofs1 = locate_dofs_topological(P2, facetdim, bndry_facets1)

bcs = [dirichletbc(u_bc_0, bdofs0), dirichletbc(u_bc_1, bdofs1)]

u, p = Function(P2), Function(P1)
du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)


def ε(u):
    return ufl.sym(ufl.grad(u))

def η(u):
    n = 3.0
    return (inner(ε(u), ε(u)) / 2 + 1.e-8)**((1 - n) / (2 * n))



f = Constant(mesh, (PETSc.ScalarType(0.0), PETSc.ScalarType(0)))
# #%%

# F = η(u)*inner(ε(u), ε(v)) * dx + inner(p, ufl.div(v)) * dx - inner(f, v) * dx +\
#         inner(ufl.div(u), q) * dx
# TH = mixed_element([P2, P1])
# W = functionspace(mesh, TH)

# w = Function(W)


# import dolfinx
# # Solve for (u,p)
# problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=bcs)
# solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
# n, converged = solver.solve(w)



# #%%
F = [η(u)*inner(ε(u), ε(v)) * dx + inner(p, ufl.div(v)) * dx - inner(f, v) * dx,
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

problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
snes.setFunction(problem.F_nest, Fvec)
snes.setJacobian(problem.J_nest, J=Jmat, P=Pmat)

u.interpolate(initial_guess_u)
p.interpolate(initial_guess_p)
x = create_vector_nest(F)
# for x1_soln_pair in zip(x.getNestSubVecs(), (u, p)):
#     x1_sub, soln_sub = x1_soln_pair
#     soln_sub.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#     soln_sub.vector.copy(result=x1_sub)
#     x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# x.set(0.0)
assert x.getType() == "nest"
for x_soln_pair in zip(x.getNestSubVecs(), (u, p)):
    x_sub, soln_sub = x_soln_pair
    soln_sub.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    soln_sub.x.petsc_vec.copy(result=x_sub)
    x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

snes.solve(None, x)
assert snes.getKSP().getConvergedReason() > 0

Fnorm = Fvec.norm()
Jnorm = nest_matrix_norm(Jmat)
xnorm = x.norm()

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "out_stokes/velocitynonlinear.xdmf", "w") as ufile_xdmf:
        u.x.scatter_forward()
        P1 = element(
            "Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type
        )
        u1 = Function(functionspace(mesh, P1))
        u1.interpolate(u)
        ufile_xdmf.write_mesh(mesh)
        ufile_xdmf.write_function(u1)

# %%
