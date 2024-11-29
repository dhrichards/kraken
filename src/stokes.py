import dolfinx.nls.petsc
import numpy as np
from dolfinx import fem, default_real_type, default_scalar_type
import basix.ufl as bufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from phasefield import degradation, degraded_pressure
from elasticity import water_pressure
from nonlinear import NonlinearPDE_SNESProblem
from material import MaterialProperties
from common import *


def solve(msh, bc_func, vh, material, dt, d=None, u=None, p=None):

    C1 = material.C1; C2 = material.C2
    ρratio = material.ρratio

    P2_el = bufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
    P1_el = bufl.element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)

    P2 = fem.functionspace(msh, P2_el)
    P1 = fem.functionspace(msh, P1_el)

    bcs = bc_func(P2)

    if u is None:
        u = fem.Function(P2, name="velocity")
    if p is None:
        p = fem.Function(P1, name="pressure")
    du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    if d is None:
        d = fem.Constant(msh, default_scalar_type(0.0))

    def η(u):
        return viscosity(u, material.n, 1.e-8)
    
    def hat(p):
        return degraded_pressure(p, d)
    
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))

    # Outward-pointing unit normal to the boundary  
    n = ufl.FacetNormal(msh)           

    # Surface measure
    ds = ufl.Measure("ds", domain=msh)

    # Water pressure
    def pw(u):
        return water_pressure(msh,u*dt + vh)
    

    # Phase field changes
    g = degradation(d)

    
    F = [((1/C2)*g*η(u)*ufl.inner(ε(u), ε(v)) \
        + ufl.inner(hat(-p), ufl.div(v))\
        - C1 * g * ufl.inner(f, v) \
        + C1 * pw(u) * ufl.inner(ufl.grad(g), v)) * ufl.dx \
        + C1 * g * pw(u) * ufl.inner(n, v) * ds,
        - ufl.inner(ufl.div(u), q) * ufl.dx ]
    
    # F = [((1/C2)*η(u)*ufl.inner(ufl.grad(u), ufl.grad(v))  \
    #     - ufl.inner(p, ufl.div(v))  \
    #     - C1 * ufl.inner(f, v)) * ufl.dx \
    #     + C1*pw(u)*ufl.inner(n, v) * ds,
    #     - ufl.inner(ufl.div(u), q) * ufl.dx ]
    

    J = get_jacobian(F,u,p,du,dp)
    P = get_preconditioner(J, u, dp, q, η) 


    return _nested_solve(F, J, P, u, p, bcs)


def solve_no_damage(mesh, bc_func, material, dt, u=None, p=None):

    C1 = material.C1; C2 = material.C2
    ρw = material.ρw; ρi = material.ρi

    P2_el = bufl.element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
    P1_el = bufl.element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)

    P2 = fem.functionspace(mesh, P2_el)
    P1 = fem.functionspace(mesh, P1_el)

    bcs = bc_func(P2)

    if u is None:
        u = fem.Function(P2)
    if p is None:
        p = fem.Function(P1)

    du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    def η(u):
        return viscosity(u, material.n, 1.e-8)
    

    if mesh.geometry.dim == 2:
        f = fem.Constant(mesh, default_scalar_type((0, -ρi/ρw)))
    else:
        f = fem.Constant(mesh, default_scalar_type((0, 0, -ρi/ρw)))


    # Outward-pointing unit normal to the boundary  
    n = ufl.FacetNormal(mesh)           

    # Surface measure
    ds = ufl.Measure("ds", domain=mesh)

    # Water pressure
    # pw = water_pressure(mesh,vh,material)
    def pw(u):
        return water_pressure(mesh,u*dt,material)
    # pw = water_pressure_static(mesh)

    # Create nullspace
    # c = fem.Function(P2)
    # c.interpolate(lambda x: np.array([[0],[1]])) # Constraint $v[1]$ averaged to zero.
    # c2 = c.x.petsc_vec
    # c2.scale(1 / c2.norm())

    # nullspace = PETSc.NullSpace().create(vectors=[c2], comm=mesh.comm)



    
    F = [(1/C2)*η(u)*ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        - ufl.inner(p, ufl.div(v)) * ufl.dx \
        - C1 * ufl.inner(f, v) * ufl.dx \
        + C1*pw(u)*ufl.inner(n, v) * ds,
        - ufl.inner(ufl.div(u), q) * ufl.dx ]
    

    J = get_jacobian(F,u,p,du,dp)
    P = get_preconditioner(J, u, dp, q, η) 
    
    
    return _nested_solve(F, J, P, u, p, bcs)
    

def get_jacobian(F,u,p,du,dp):
    return [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
            [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)]]

def get_preconditioner(J, u, dp, q, η):
    return [[J[0][0], None],
            [None, (2 * η(u))**-1 * dp * q * ufl.dx]]


def _block_solve(F, J , P, u, p, bcs, V, Q):
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
    snes.setTolerances(rtol=1.0e-15, max_it=20)
    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs=bcs, P=P)

    snes.setFunction(problem.F_block,
                        dolfinx.fem.petsc.create_vector_block(F))
    snes.setJacobian(problem.J_block,
                        J=dolfinx.fem.petsc.create_matrix_block(J),
                        P=None)
    x = dolfinx.fem.petsc.create_vector_block(F)


    snes.solve(None, x)
    assert snes.getKSP().getConvergedReason() > 0

    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p


def _nested_solve(F, J, P, u, p, bcs):
    F, J, P = fem.form(F), fem.form(J), fem.form(P)


    Jmat = fem.petsc.create_matrix_nest(J)
    Pmat = fem.petsc.create_matrix_nest(P)
    Fvec = fem.petsc.create_vector_nest(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-15, max_it=10)
    nested_IS = Jmat.getNestISs()
    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-12)
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
    snes.solve(None, x)
    assert snes.getKSP().getConvergedReason() > 0

    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p






