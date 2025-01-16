#%%

from dolfinx import mesh, default_scalar_type
from mpi4py import MPI
import numpy as np
from kraken.material import Material_no_uc
from kraken.boundaryconditions import get_zero_bc, get_bc
from kraken import energybased as eb, stokes, utilities


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length)


true_length = 1.0
true_height = 0.3

material = Material_no_uc(g=1e-9, E=100, ν=0.3, ρi=0.0)
material.L = 1.0
material.l = 0.1
material.ψcritstar = 0.0

cell_size = material.l/6


nondim_length = true_length/material.L
nondim_height = true_height/material.L

nx = int(nondim_length/cell_size)
ny = int(nondim_height/cell_size)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([nondim_length, nondim_height])],
                            [nx,ny], mesh.CellType.quadrilateral)


# material.set_l_from_mesh(msh)
# 
ubc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                    get_bc(V.sub(0), right_boundary, default_scalar_type(1.0)) ]

dbc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                 get_zero_bc(V, right_boundary, default_scalar_type)]

# log.set_log_level(log.LogLevel.INFO)

vh, dh = eb.fixed_point(msh, [ubc, dbc], material)
uh, ph = stokes.solve(msh, ubc, vh, material, 1.0, dh)
# vh, dh = pf.minimisation(msh, [ubc, dbc], material)
# vh, dh = monolithic.solve(msh, ubc, material)


# utilities.plot_damage_state(vh,dh)
utilities.write_vtk("outputs/newfrac.pvd", msh, \
                    [vh,dh,uh], \
                    ["v","d","u"])
