import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem, io


L, H = 0.1, 0.2
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([L, H])],
                            [20, 10], mesh.CellType.triangle)



E0 = 70e3
E1 = 20e3
eta1 = 1e3
ν = 0
dt = 0# time incremen
sigc = 100. # imposed creep stress
epsr = 1e-3 # imposed relaxation strain


def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

def bottom(x):
    return np.isclose(x[1], 0)

def top(x):
    return np.isclose(x[1], H)

V_el = basix.ufl.element("CG", msh.basix_cell(), 1, shape=(2,))
Q_el = basix.ufl.element("DG", msh.basix_cell(), 0, shape=(2,2))

W_el = basix.ufl.mixed_element([V_el, Q_el])

V = fem.functionspace(msh, V_el)
Q = fem.functionspace(msh, Q_el)
W = fem.functionspace(msh, W_el)


w = fem.Function(W, name="Variables at current step")
(u, epsv) = fem.split(w)

w_old = fem.Function(W, name="Variables at previous step")
(u_old, epsv_old) = split(w_old)
w_ = ufl.TestFunction(W)
(u_, epsv_) = split(w_)
dw = ufl.TrialFunction(W)


def ε(u):
    return ufl.sym(ufl.grad(u))

def dotC(ε):
    return ν/(1+ν)/(1-ν)*ufl.tr(ε)*ufl.Identity(2) + 1/(1+ν)*ε

def strain_energy(ε, εv):
    εe = ε - εv
    return 0.5*(E0*ufl.inner(ε,dotC(ε)) + E1*ufl.inner(εe, dotC(εe)))



def dissipation_potential(depsv):
    return 0.5*eta1*ufl.inner(depsv, depsv)

Traction = 0.0
incremental_potential = strain_energy(ε(w), ε(u))*ufl.dx \
                        + dt*dissipation_potential((ε(u)-ε(u_old))/dt)*ufl.dx
F = ufl.derivative(incremental_potential, w, w_)
form = replace(F, {w: dw})

