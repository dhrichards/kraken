#%%
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import kraken
from kraken import boundaryconditions as bc
import kraken.mainclass as mc
import kraken.material



L = 5e3 # length of 1 wavelength
H = 1e3
repetitions = 8
Lmax = 4

nz = 10
nx = int(nz*L*repetitions/H)

def top(x):
    return H

def bottom(x):
    return 0.5*H*np.sin(x*2*np.pi/L)

def bottom_boundary(x):
    r = -x[1] + bottom(x[0])
    return np.isclose(r,0.0)

def left_boundary(x):
    return np.isclose(x[0],-L*repetitions//2)

def right_boundary(x):
    return np.isclose(x[0],L*repetitions//2)

def top_boundary(x):
    return np.isclose(x[1],H)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([-L*repetitions//2, 0]), np.array([L*repetitions//2, 1])],
                            [nx,nz], mesh.CellType.quadrilateral)


def warp_mesh(msh,top,bot):
    x = msh.geometry.x[:,0]
    z = msh.geometry.x[:,1]
    zs = top(x)
    zb = bot(x)
    znew = zb + z*(zs-zb)
    xynewcoor = np.array([x, znew]).transpose()
    msh.geometry.x[:,:2] = xynewcoor



warp_mesh(msh,top,bottom)







material = kraken.material.Material_no_uc()
material.τ = 1.0
material.L = 1.0
material.A = 1e-16
material.slope_angle = 0.5

bc_u = lambda V: [bc.get_zero_bc(V, bottom_boundary),
                  bc.get_zero_bc(V, left_boundary),
                  bc.get_zero_bc(V, right_boundary)]
bc_f = lambda V: []
bc_z = lambda V: []

model = mc.viscoelastic_damage(msh,[bc_z,bc_u,bc_f],material,0.0)
model.stokes.pw = lambda u: 0.0
model.solve_stokes()

kraken.utilities.write_xdmf("outputs/ISMIPB.xdmf", msh, [model.u], "u")

    

