#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.parameters import Params_no_uc, Params_with_uc
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken.utilities as utilities
import kraken.oneclass as oc

radius = 52
thickness = 42
notch = 26

bottom_roller_x = 40
bottom_roller_z = 0


top_roller_z = 54

roller_radius = 3.0






def bottom_roller(x):
    r = np.sqrt((x[0]-bottom_roller_x)**2 + (x[1]-bottom_roller_z)**2)
    return r < roller_radius


def top_roller(x):
    r = np.sqrt((x[0])**2 + (x[1]-top_roller_z)**2)
    return r < roller_radius

def left_boundary(x):
    return np.isclose(x[0], 0)

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



# material = Material_no_uc()
material = Params_no_uc()
material.L = 1e-3
material.ψcrit = 0.0
material.l = 1.5e-3
material.g = 1e-9



filename = "experiment2d.xdmf"

# msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
with io.XDMFFile(MPI.COMM_WORLD,"meshes/" + filename,"r") as infile:
    msh = infile.read_mesh()

no_bc = lambda V: []
model = oc.viscoelastic_damage(msh, [no_bc,no_bc,no_bc], material, 
                               dt = 1)#,g = lambda d: mf.degradation_Lo2023(d,0.05))

model.p_ext = lambda u: 0.0


# model.free_energy_plus = es.free_energy
model.w = lambda d: d
model.bounded = True
model.setup_elastic()
model.setup_damage()



#%%


disps = np.linspace(0,0.03,100)
for i in range(len(disps)):


    bottom_roller_x = np.sqrt(40**2 + (radius-disps[i])**2 - radius**2)

    disp_bc = lambda V: [bc.get_zero_bc(V.sub(1), bottom_roller),
                            bc.get_bc(V.sub(1), top_roller, -disps[i]),
                            bc.get_zero_bc(V.sub(0), left_boundary)]
    model.bc_v = disp_bc(model.V)
    model.setup_elastic()

    model.fixed_point_simple(max_its=300)


    utilities.write_xdmf(path +"experiment" + str(i) + ".xdmf", msh, \
                    [model.v,model.d],\
                    ["v","d"], t=disps[i])

