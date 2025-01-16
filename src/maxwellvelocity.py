#%%

from dolfinx import mesh, io, default_scalar_type
from mpi4py import MPI
import numpy as np
from kraken import elasticity, stokes, utilities
from kraken.material import MaterialProperties
from kraken.boundaryconditions import get_zero_bc


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)


true_length = 16e3
true_height = 300

material = MaterialProperties()

nondim_length = true_length/material.L
nondim_height = true_height/material.L


Hw = material.ρi/material.ρw*nondim_height

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
                            [100,20], mesh.CellType.triangle)


material.set_l_from_mesh(msh)

dt = 1e-9
# 
clamped_both = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type),
                            get_zero_bc(V, right_boundary, default_scalar_type)]

clamped_bc = lambda V: [get_zero_bc(V, left_boundary, default_scalar_type)]
symm_bc = lambda V: [get_zero_bc(V.sub(0), left_boundary, default_scalar_type)]
no_bc = lambda V: []

bc = symm_bc
# vh = elasticity.solve(msh,material,bc,0.0)

for i in range(1):
    print(i)
    vh = elasticity.solve(msh, bc, material)

    # utilities.move_mesh(msh,vh,material.uc/material.L)

    uh, ph = stokes.solve(msh, bc, vh, material, dt)

    with io.VTKFile(MPI.COMM_WORLD, "outputs/displacement.pvd","w") as file:
        file.write_mesh(msh,t=i*dt)
        file.write_function([vh],t=i*dt)
    
    with io.VTKFile(MPI.COMM_WORLD, "outputs/velocity.pvd","w") as file:
        file.write_mesh(msh, t=i*dt)
        file.write_function([uh],t=i*dt)

    utilities.move_mesh(msh, uh, dt * material.uc / material.L)












