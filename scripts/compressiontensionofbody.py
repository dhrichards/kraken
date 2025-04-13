#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
from kraken.material import Material_no_uc, Material_with_uc
import kraken.boundaryconditions as bc
import kraken.utilities as utilities
import kraken.mainclass as mc
import kraken.oneclass as oc
from kraken.numerics import maths_functions as mf


d = 50
h = 100
l = 2
cell_size = l/2


dt = 1


nx = int(d/cell_size)
nz = int(h/cell_size)


def bottom(x):
    return np.isclose(x[1], 0) 
def top(x):
    return np.isclose(x[1], h)





material = Material_with_uc()
material.L = 1.0
material.uc = 1.0
material.ρi = 0.0
material.ρw = 1.0
material.g = 1e-12

material.E = 30e3
material.ν = 0.2
material.ψcrit = 0.0
material.l = l
material.Gc = 1e-3


msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([d,h])],
                            [nx,nz], mesh.CellType.quadrilateral)


bc_v = lambda V: [bc.get_zero_bc(V, bottom),
                  bc.get_zero_bc(V.sub(0), top),
                    bc.get_bc(V.sub(1), top, 0.0)]

bc_d = lambda V: [bc.get_zero_bc(V, bottom),
                  bc.get_zero_bc(V, top)]                     

model = oc.viscoelastic_damage(msh, [bc_v,bc_v,bc_d], material, 1.0)

# model.free_energy_plus = mf.free_energy_plus_basic
# model.bounded = True
# model.w = lambda d: d

#%%
# ubar = 2.5e-4
ubar = -25e-4
if ubar>0:
    filename = "traction"
else:
    filename = "compression"

model.setup_damage()
for t in range(200):
    if MPI.COMM_WORLD.rank == 0:
        print("Time step: ",t)
    model.bc_v = [bc.get_zero_bc(model.V, bottom),
                        bc.get_zero_bc(model.V.sub(0), top),
                        bc.get_bc(model.V.sub(1), top, ubar*t)]

    model.setup_elastic()
    model.fixed_point_simple(max_its=200)
    utilities.write_xdmf("outputs/" +filename + str(t) + ".xdmf",model.msh,
                        [model.v,model.d,model.Hprev],["v","d","H"],t=t)


