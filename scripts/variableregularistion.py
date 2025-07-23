#%%

from dolfinx import fem, mesh
import ufl
from petsc4py import PETSc
from mpi4py import MPI
import basix.ufl as bufl
import kraken.boundaryconditions as bc
import kraken.utilities as utilities
import numpy as np


L = 2.0
a_aim = 0.2
h_aim = a_aim/5
nx = int(L/h_aim)

h = L/nx
# get a as a multiple of grid spacing
a = int(a_aim/h)*h



w = h # crack width

msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-L/2, -L/2]), np.array([L/2, L/2])],
                            [nx,nx], mesh.CellType.quadrilateral)


D = fem.functionspace(msh, ("Lagrange", 1))

def crack(x):
    return (x[0]>=-1.001*a)*(x[0]<=1.001*a)*(x[1]>-1e-6)*(x[1]<1.001*w)


bc_d = bc.internal_bc(D, crack, 1.0)



d = ufl.TrialFunction(D)
v = ufl.TestFunction(D)

lx = 0.4
ly = 0.1

# K = ufl.as_tensor([[lx**2, 0], [0, ly**2]], (2, 2))
K = ufl.as_matrix([[lx**2, 0], [0, ly**2]])
# K = lx**2 * ufl.Identity(msh.geometry.dim)


a = (ufl.inner(d,v) + ufl.inner(ufl.dot(K,(ufl.grad(d))), ufl.grad(v)))*ufl.dx 
L = fem.Constant(msh, 0.0)*v*ufl.dx




damage_problem = fem.petsc.LinearProblem(a, L, bcs=[bc_d],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


dh = damage_problem.solve()


utilities.write_xdmf("variable_l.xdmf", msh, [dh], ["damage"])