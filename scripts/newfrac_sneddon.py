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

import sys
sys.path.append("../utils/")

import dolfinx.fem.petsc

import pyvista
def plot_damage_state(u, d):
    """
    Plot the displacement and damage field with pyvista
    """

    mesh = u.function_space.mesh



    topology, cell_types, geometry = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    plotter.view_xy()
    plotter.add_axes()
    plotter.set_scale(5,5)

    plotter = pyvista.Plotter(
        title="Damage state", window_size=[800, 300], shape=(1, 2)
    )

    topology, cell_types, x = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    
    plotter.subplot(0, 0)
    plotter.add_text("Displacement", font_size=11)
    vals = np.zeros((x.shape[0], 3))
    vals[:,:len(u)] = u.x.array.reshape((x.shape[0], len(u)))
    grid["u"] = vals
    warped = grid.warp_by_vector("u", factor=0.1)
    actor_1 = plotter.add_mesh(warped, show_edges=False)
    plotter.view_xy()

    plotter.subplot(0, 1)

    plotter.add_text("Damage", font_size=11)

    grid.point_data["alpha"] = d.x.array
    grid.set_active_scalars("alpha")
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, clim=[0, 1])
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
       plotter.show()



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


L = 1.0
a_aim = 0.2
h_aim = a_aim/10
nx = int(2*L/h_aim)

h = 2*L/nx
ell_ = 0.05
# get a as a multiple of grid spacing
a_crack = int(a_aim/h)*h



width_crack = h # crack width



comm = MPI.COMM_WORLD
domain = mesh.create_rectangle(
    comm, [(-L, -L), (L, L)], [nx, nx], cell_type=mesh.CellType.quadrilateral
)
ndim = domain.geometry.dim
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_alpha = fem.functionspace(domain, ("Lagrange", 1))

# Define the state
u = fem.Function(V_u, name="Displacement")
alpha = fem.Function(V_alpha, name="Damage")

state = {"u": u, "alpha": alpha}

# need upper/lower bound for the damage field
alpha_lb = fem.Function(V_alpha, name="Lower bound")
alpha_ub = fem.Function(V_alpha, name="Upper bound")
alpha_ub.x.array[:] = 1
alpha_lb.x.array[:] = 0

# Measures
dx = ufl.Measure("dx",domain=domain)
ds = ufl.Measure("ds",domain=domain)


def bottom(x):
    return np.isclose(x[1], -L)

def top(x):
    return np.isclose(x[1], L)

def right(x):
    return np.isclose(x[0], L)

def left(x):
    return np.isclose(x[0], -L)

def boundary(x):
    return np.isclose(x[0], -L) | \
        np.isclose(x[0], L) | \
        np.isclose(x[1], -L) | \
        np.isclose(x[1], L)


fdim = domain.topology.dim-1

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)
left_boundary_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, left_facets)
right_boundary_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, right_facets)
bottom_boundary_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, bottom_facets)
top_boundary_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, top_facets)


# u_D = fem.Constant(domain,PETSc.ScalarType(0.,0.0))
# bc_u_left = fem.dirichletbc(u_D, left_boundary_dofs_ux, V_u)
# bc_u_right = fem.dirichletbc(u_D, right_boundary_dofs_ux, V_u)
# bc_u_bottom = fem.dirichletbc(u_D, bottom_boundary_dofs_uy, V_u)
# bc_u_top = fem.dirichletbc(u_D, top_boundary_dofs_uy, V_u)


import boundarycondtions as bc

bc_all = bc.get_zero_bc(V_u, boundary)
bcs_u = [bc_all]




def crack(x):
    return (x[0]>=-1.001*a_crack)*(x[0]<=1.001*a_crack)*(x[1]>-1e-6)*(x[1]<1.001*width_crack)

bcs_alpha = [bc.internal_bc(V_alpha, crack, 1.0)]
# left_boundary_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, left_facets)
# right_boundary_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, right_facets)
# bc_alpha_left = fem.dirichletbc(0.0, left_boundary_dofs_alpha, V_alpha)
# bc_alpha_right = fem.dirichletbc(0.0, right_boundary_dofs_alpha, V_alpha)

# bcs_alpha = [bc_alpha_left,bc_alpha_right]
p_ = 0.1
p= fem.Function(V_alpha)
p.x.array[:] = 0.1
E_, nu_ = 1.0, 0.2
Gc_ = 1.0


p_c = np.sqrt(Gc_*E_/(np.pi*a_crack*(1-nu_**2)))
print("p_c = ",p_c)
# E, nu = fem.Constant(domain, PETSc.ScalarType(E_)), fem.Constant(domain, PETSc.ScalarType(nu_))
# Gc = fem.Constant(domain, PETSc.ScalarType(Gc_))
# ell = fem.Constant(domain, PETSc.ScalarType(ell_))

E = E_; nu = nu_
Gc = Gc_
ell = ell_

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return alpha**2

def a(alpha, k_ell=1.e-6):
    """Stiffness modulation as a function of the damage """
    return (1 - alpha) ** 2 + k_ell

def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))

def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    mu    = E / (2.0 * (1.0 + nu))
    # lmbda = E * nu / (1.0 - nu ** 2)
    lmbda = E*nu/((1+nu)*(1-2*nu))
    return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(alpha) * sigma_0(u)

def free_energy(u):
    λ = E*nu/((1+nu)*(1-2*nu))
    μ = E / (2.0 * (1.0 + nu))

    return 0.5*(λ*ufl.tr(eps(u))**2 + 2*μ*ufl.inner(eps(u), eps(u)))


import sympy 
z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
print("c_w = ",c_w)

c_1w = sympy.integrate(sympy.sqrt(1/w(z)),(z,0,1))
print("c_1/w = ",c_1w)

tmp = 2*(sympy.diff(w(z),z)/sympy.diff(1/a(z),z)).subs({"z":0})
sigma_c = sympy.sqrt(tmp * Gc_ * E_ / (c_w * ell_))
print("sigma_c = %2.3f"%sigma_c)

eps_c = float(sigma_c/E_)
print("eps_c = %2.3f"%eps_c)


f = fem.Constant(domain,PETSc.ScalarType((0.,0.)))
# elastic_energy = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx 
elastic_energy = a(alpha)*free_energy(u) * dx
dissipated_energy = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.inner(ufl.grad(alpha), ufl.grad(alpha))) * dx
external_work = p*ufl.inner(ufl.grad(a(alpha)), u) * dx 
total_energy = elastic_energy + dissipated_energy - external_work



E_u = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
E_u_u = ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
elastic_problem = SNESProblem(E_u, u, bcs_u)

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


E_alpha = ufl.derivative(elastic_energy + dissipated_energy,alpha,ufl.TestFunction(V_alpha))
E_alpha_alpha = ufl.derivative(E_alpha,alpha,ufl.TrialFunction(V_alpha))
damage_problem = SNESProblem(E_alpha, alpha, bcs_alpha,J=E_alpha_alpha)

b_alpha = la.create_petsc_vector(V_alpha.dofmap.index_map, V_alpha.dofmap.index_map_bs)
J_alpha = fem.petsc.create_matrix(damage_problem.a)
# Create Newton solver and solve
solver_alpha_snes = PETSc.SNES().create()
solver_alpha_snes.setType("vinewtonrsls")
solver_alpha_snes.setFunction(damage_problem.F, b_alpha)
solver_alpha_snes.setJacobian(damage_problem.J, J_alpha)
solver_alpha_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_alpha_snes.getKSP().setType("preonly")
solver_alpha_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_alpha_snes.getKSP().getPC().setType("lu")
# We set the bound (Note: they are passed as reference and not as values)
solver_alpha_snes.setVariableBounds(alpha_lb.x.petsc_vec,alpha_ub.x.petsc_vec)


alt_min_parameters = {"atol": 1.e-8, "max_iter": 100}

def simple_monitor(state, iteration, error_L2):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Iteration: {iteration:3d}, Error: {error_L2:3.4e}")
    
def alternate_minimization(state,parameters=alt_min_parameters,monitor=None):
    L2_old = 0
    u = state["u"]
    alpha = state["alpha"]
    
    alpha_old = fem.Function(alpha.function_space)
    alpha.x.petsc_vec.copy(result=alpha_old.x.petsc_vec)
    
    for iteration in range(parameters["max_iter"]):
                              
        # solve displacement
        solver_u_snes.solve(None, u.x.petsc_vec)
        
        # solve damage
        solver_alpha_snes.solve(None, alpha.x.petsc_vec)
        
        # check error and update
        L2_ = ufl.inner(alpha,alpha)*ufl.dx
        L2_rank = fem.assemble_scalar(fem.form(L2_))
        L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))
        error_L2 = np.abs(L2 - L2_old)
        
        alpha.x.petsc_vec.copy(alpha_old.x.petsc_vec)
        
        if monitor is not None:
            monitor(state, iteration, error_L2)
        L2_old = L2
                                 
        if error_L2 <= parameters["atol"]:
            break
    else:
        pass #raise RuntimeError(f"Could not converge after {iteration:3d} iteration, error {error_L2:3.4e}") 
    
    return (error_L2, iteration)


alpha.x.array[:] = 00
    
# alternate_minimization(state,parameters=alt_min_parameters,monitor=simple_monitor)
# plot_damage_state(u, alpha)

# aeff_yoi = a*(1 + (π*l/4) / (a*(h/(2*l) + 1)))
aeff_jakub = a_crack + np.pi*ell_/4

def umax(a):
    return 2*p_*a*(1-nu_**2)/E_

#%%
ps = np.linspace(1,2,50)
for i in range(50):
    print(i)
    
    p.x.array[:] = ps[i]
    alternate_minimization(state,parameters=alt_min_parameters,monitor=simple_monitor)

    utilities.write_xdmf("newfrac" + str(i) + ".xdmf",domain,
                        [u,alpha],["u","d"],t=ps[i])


# v_max = MPI.COMM_WORLD.allreduce(np.max(u.x.array), op=MPI.MAX)
# print("umax theory = ",umax(aeff_jakub))
# print("model:",v_max)