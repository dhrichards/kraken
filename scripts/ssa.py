#%%
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken.numerics import solvers, invariants
from kraken import utilities
from kraken import boundaryconditions as bc
from petsc4py import PETSc

L = 50e3
msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, L]], [50, 50], mesh.CellType.triangle)
# msh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, mesh.CellType.triangle)

def top_boundary(x):
    return np.isclose(x[1], L)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

u_bc = lambda V: [bc.get_zero_bc(V, bottom_boundary),
                           bc.get_bc(V, top_boundary, np.array([0,10.0]))]
        
u_el = bufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
h_el = bufl.element("CG", msh.basix_cell(), 1)

U = fem.functionspace(msh, u_el)
H = fem.functionspace(msh, h_el)



B = 100
n = 3.0
p = 1 + 1/n

ε = lambda u: ufl.sym(ufl.grad(u))
η = lambda ε: (0.5*ufl.inner(ε, ε) + 0.5*ufl.tr(ε)**2 + 1e-9)**((p-2)/2)

u = fem.Function(U, name="velocity")
h = fem.Function(H, name="thickness")
h.x.array[:] = 300.0
b = 0
s = h + b


N = η(ε(u)) * ε(u)
T = N + ufl.tr(N) * ufl.Identity(2)

v = ufl.TestFunction(U)
n = ufl.FacetNormal(msh)
ρi = 9.138e-19
g = 9.7692e15
# g = 9.81
δ = 0.9

F = B*h*ufl.inner(T, ε(v)) * ufl.dx \
    - ρi * g * h * ufl.inner(ufl.grad(s), v) * ufl.dx \
    + 0.5*δ*ρi*g*h**2*ufl.dot(n,v)*ufl.ds

J = ufl.derivative(F, u, ufl.TrialFunction(U))


problem = solvers.SNESProblem(F, u, bcs=u_bc(U))

solver = PETSc.SNES().create(MPI.COMM_WORLD)
solver.setTolerances(rtol=1.0e-8, max_it=50, atol=1e-10)
solver.getKSP().setType("preonly")
# solver.getKSP().setTolerances(rtol=1.0e-7)
solver.getKSP().getPC().setType("lu")
solver.getKSP().getPC().setFactorSolverType("mumps")


solver.setFunction(problem.F, fem.petsc.create_vector(fem.form(F)))
solver.setJacobian(problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)

solver.solve(None, u.x.petsc_vec)


# print(u.x.array)
λ,Q = invariants.eigenstate(N)
λplus = [ufl.max_value(lam, 0.0) for lam in λ]

σplus = ufl.diag(ufl.as_vector(λplus))


utilities.write_xdmf("../outputs/ssa.xdmf",
                      msh, [u, h, λ[1]],
                      ["u", "h", "λ"],
                      t=0)