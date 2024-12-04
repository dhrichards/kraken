#%%

import matplotlib.pyplot as plt
import numpy as np

import dolfinx
from dolfinx import mesh, fem, plot, io, la
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import pyvista
from pyvista.utilities.xvfb import start_xvfb
start_xvfb(wait=0.5)
import utilities

import dolfinx.fem.petsc
import phasefield
from common import *


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



L = 1.; H = 0.3
l_ = 0.1
cell_size = l_/6

nx = int(L/cell_size)
ny = int(H/cell_size)

comm = MPI.COMM_WORLD
domain = mesh.create_rectangle(
    comm, [(0.0, 0.0), (L, H)], [nx, ny], cell_type=mesh.CellType.quadrilateral
)
ndim = domain.geometry.dim




V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_d = fem.functionspace(domain, ("Lagrange", 1))

# Define the state
u = fem.Function(V_u, name="Displacement")
d = fem.Function(V_d, name="Damage")

state = {"u": u, "d": d}

# need upper/lower bound for the damage field
d_lb = fem.Function(V_d, name="Lower bound")
d_ub = fem.Function(V_d, name="Upper bound")
d_ub.x.array[:] = 1
d_lb.x.array[:] = 0

# Measures
dx = ufl.Measure("dx",domain=domain)
ds = ufl.Measure("ds",domain=domain)



def bottom(x):
    return np.isclose(x[1], 0.0)

def top(x):
    return np.isclose(x[1], H)

def right(x):
    return np.isclose(x[0], L)

def left(x):
    return np.isclose(x[0], 0.0)

fdim = domain.topology.dim-1

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)
left_boundary_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, left_facets)
right_boundary_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, right_facets)
bottom_boundary_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, bottom_facets)
top_boundary_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, top_facets)


u_D = fem.Constant(domain,PETSc.ScalarType(1.))
bc_u_left = fem.dirichletbc(0.0, left_boundary_dofs_ux, V_u.sub(0))
bc_u_right = fem.dirichletbc(u_D, right_boundary_dofs_ux, V_u.sub(0))
bc_u_bottom = fem.dirichletbc(0.0, bottom_boundary_dofs_uy, V_u.sub(1))
bc_u_top = fem.dirichletbc(0.0, top_boundary_dofs_uy, V_u.sub(1))
bcs_u = [bc_u_left,bc_u_right]#,bc_u_bottom,bc_u_top]

left_boundary_dofs_d = fem.locate_dofs_topological(V_d, fdim, left_facets)
right_boundary_dofs_d = fem.locate_dofs_topological(V_d, fdim, right_facets)
bc_d_left = fem.dirichletbc(0.0, left_boundary_dofs_d, V_d)
bc_d_right = fem.dirichletbc(0.0, right_boundary_dofs_d, V_d)

bcs_d = [bc_d_left,bc_d_right]



E, ν = fem.Constant(domain, PETSc.ScalarType(100.0)), fem.Constant(domain, PETSc.ScalarType(0.3))
Gc = fem.Constant(domain, PETSc.ScalarType(1.0))
l = fem.Constant(domain, PETSc.ScalarType(l_)) 

def w(d):
    """Dissipated energy function as a function of the damage """
    return d**2

def g(d, k=1.e-6):
    """Stiffness modulation as a function of the damage """
    return (1 - d) ** 2 + k

def ε(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))

def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    mu    = E / (2.0 * (1.0 + ν))
    lmbda = E * ν / (1.0 - ν ** 2)
    return 2.0 * mu * ε(u) + lmbda * ufl.tr(ε(u)) * ufl.Identity(ndim)

def σ(u,d):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return g(d) * sigma_0(u)

def γ(d):
    return 0.5/l * (w(d) + l**2 * ufl.inner(ufl.grad(d), ufl.grad(d)))

def ψ(u,d):
    """Elastic energy as a function of the displacement and the damage"""
    return phasefield.degradation(d)*phasefield.free_energy(u,ν)


def ψ(u,d):
    """Elastic energy as a function of the displacement and the damage"""
    ψplus = phasefield.free_energy_plus(u,ν)
    ψminus = phasefield.free_energy(u,ν) - ψplus
    return phasefield.degradation(d) * ψplus + ψminus




μ = fem.Constant(domain,PETSc.ScalarType(E/(2*(1+ν))))
f = fem.Constant(domain,PETSc.ScalarType((0.,0.)))

total_energy = μ/Gc*ψ(u,d) * dx + γ(d) * dx - ufl.dot(f, u) * dx


E_u = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
v = ufl.TestFunction(V_u)
# E_u = ufl.inner(σ(u,d), ε(v)) * dx
E_u_u = ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
elastic_problem = SNESProblem(E_u, u, bcs_u, J=E_u_u)

b_u = la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
J_u = fem.petsc.create_matrix(elastic_problem.a)
# Create Newton solver and solve
solver_u_snes = PETSc.SNES().create()
solver_u_snes.setType("ksponly")
solver_u_snes.setFunction(elastic_problem.F, b_u)
solver_u_snes.setJacobian(elastic_problem.J, J_u)
solver_u_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_u_snes.getKSP().setType("preonly")
solver_u_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_u_snes.getKSP().getPC().setType("lu")
# solver_u_snes.getKSP().getPC().setFactorSolverType("mumps")


        
load = 1.
u_D.value = load
u.x.array[:] = 0
solver_u_snes.solve(None, u.x.petsc_vec)
# plot_damage_state(state,load=load)



E_d = ufl.derivative(total_energy,d,ufl.TestFunction(V_d))
E_d_d = ufl.derivative(E_d,d,ufl.TrialFunction(V_d))
damage_problem = SNESProblem(E_d, d, bcs_d,J=E_d_d)


b_d = la.create_petsc_vector(V_d.dofmap.index_map, V_d.dofmap.index_map_bs)
J_d = fem.petsc.create_matrix(damage_problem.a)
# Create Newton solver and solve
solver_d_snes = PETSc.SNES().create()
solver_d_snes.setType("vinewtonrsls")
solver_d_snes.setFunction(damage_problem.F, b_d)
solver_d_snes.setJacobian(damage_problem.J, J_d)
solver_d_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_d_snes.getKSP().setType("preonly")
solver_d_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_d_snes.getKSP().getPC().setType("lu")
# We set the bound (Note: they are passed as reference and not as values)
solver_d_snes.setVariableBounds(d_lb.x.petsc_vec,d_ub.x.petsc_vec)

solver_d_snes.solve(None, d.x.petsc_vec)
# plot_damage_state(state,load=load)

with d.x.petsc_vec.localForm() as d_local:
    d_local.set(0)

for i in range(20):
    print(f"iteration {i}")
    solver_u_snes.solve(None, u.x.petsc_vec)
    solver_d_snes.solve(None, d.x.petsc_vec)
utilities.plot_damage_state(u,d,load)

# utilities.write_vtk("outputs/newfracexample.pvd",domain,[u],["u"])
# %%
