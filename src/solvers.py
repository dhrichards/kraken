

from dolfinx.cpp.la.petsc import scatter_local_vectors
from dolfinx.fem import (Function, functionspace,
                         bcs_by_block, extract_function_spaces)
from dolfinx.fem.petsc import (apply_lifting, apply_lifting_nest,
                               assemble_matrix, assemble_matrix_block,
                               assemble_matrix_nest, assemble_vector,
                               assemble_vector_block, set_bc,
                               set_bc_nest)
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, la



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
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
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
            size_local = var.x.petsc_vec.getLocalSize()
            var.x.petsc_vec.array[:] = x_array[offset: offset + size_local]
            var.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        assemble_vector_block(F, self.L, self.a, bcs=self.bcs, x0=x, alpha=-1.0)

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





class SNESProblem:
    def __init__(self, F, u, bcs, J=None):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        if J is None:
            self.a = fem.form(ufl.derivative(F, u, du))
        else:
            self.a = fem.form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()






def block_solve(F, J , P, u, p, bcs, V, Q):
    F, J, P = fem.form(F), fem.form(J), fem.form(P)

    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(
        V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF
    )
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    ############ Direct
    # snes.setTolerances(rtol=1.0e-10, max_it=50)
    # snes.getKSP().getPC().setType("lu")
    # snes.getKSP().setTolerances(rtol=1e-8)
    # snes.getKSP().getPC().setFactorSolverType("mumps")



    #### Iterative
    snes.setTolerances(rtol=1.0e-10, max_it=100)
    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-8)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    snes.getKSP().getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")
    # snes.getKSP().getPC().setUp()
    # Pu, _ = ksp_u
    # Pu.setBlockSize(2)



    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs=bcs, P=P)

    snes.setFunction(problem.F_block,
                        fem.petsc.create_vector_block(F))
    snes.setJacobian(problem.J_block,
                        J=fem.petsc.create_matrix_block(J),
                        P=fem.petsc.create_matrix_block(P))
    x = fem.petsc.create_vector_block(F)

    return snes, x


def nested_solve(F, J, u, p, bcs, P=None):
    F, J = fem.form(F), fem.form(J)
    if P is not None:
        P = fem.form(P)


    Jmat = fem.petsc.create_matrix_nest(J)
    if P is not None:
        Pmat = fem.petsc.create_matrix_nest(P)
    else:
        Pmat = None
    Fvec = fem.petsc.create_vector_nest(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-9, max_it=200)
    nested_IS = Jmat.getNestISs()
    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-7)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

    snes.getKSP().getPC().setFieldSplitType(
                PETSc.PC.CompositeType.ADDITIVE)

    ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("hypre")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("hypre")

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs=bcs, P=P)
    snes.setFunction(problem.F_nest, Fvec)
    snes.setJacobian(problem.J_nest, J=Jmat, P=Pmat)

    # snes.setDM(nullspace)

    

    x = fem.petsc.create_vector_nest(F)

    assert x.getType() == "nest"
    for x_soln_pair in zip(x.getNestSubVecs(), (u, p)):
        x_sub, soln_sub = x_soln_pair
        soln_sub.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        soln_sub.x.petsc_vec.copy(result=x_sub)
        x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Solve nonlinear problem
  
    return snes, x






def linear_nested_solver(a, L, u, p, bcs, P):
    """Solve the Stokes problem using nest matrices and an iterative solver."""

    # Assemble nested matrix operators
    A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = fem.petsc.assemble_matrix(P[1][1], [])
    P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
    P.assemble()

    A00 = A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)

    P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)
    P00.setOption(PETSc.Mat.Option.SPD, True)
    P11.setOption(PETSc.Mat.Option.SPD, True)

    # Assemble right-hand side vector
    b = fem.petsc.assemble_vector_nest(L)

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are share across
    # parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)

    # The pressure field is determined only up to a constant. We supply
    # a vector that spans the nullspace to the solver, and any component
    # of the solution in this direction will be eliminated during the
    # solution process.
    # null_vec = fem.petsc.create_vector_nest(L)

    # # Set velocity part to zero and the pressure part to a non-zero
    # # constant
    # null_vecs = null_vec.getNestSubVecs()
    # null_vecs[0].set(0.0), null_vecs[1].set(1.0)

    # # Normalize the vector that spans the nullspace, create a nullspace
    # # object, and attach it to the matrix
    # null_vec.normalize()
    # nsp = PETSc.NullSpace().create(vectors=[null_vec])
    # assert nsp.test(A)
    # A.setNullSpace(nsp)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-8)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Create finite element {py:class}`Function <dolfinx.fem.Function>`s
    # for the velocity (on the space `V`) and for the pressure (on the
    # space `Q`). The vectors for `u` and `p` are combined to form a
    # nested vector and the system is solved.
    x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(u.x), la.create_petsc_vector_wrap(p.x)])
    
    return ksp, x, b



def linear_block_solver(a, L, u, p, bcs, P, V, Q):

    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    P = assemble_matrix_block(P, bcs=bcs)
    P.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # Set the nullspace for pressure (since pressure is determined only
    # up to a constant)
    null_vec = A.createVecLeft()
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A)
    A.setNullSpace(nsp)

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Set the solver type to MUMPS (LU solver) and configure MUMPS to
    # handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")

    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    x = A.createVecLeft()
    ksp.solve(b, x)

    # Create Functions and scatter x solution
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

