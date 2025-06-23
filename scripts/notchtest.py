#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.parameters import Params_no_uc, Params_with_uc
import kraken.boundaryconditions as bc_bottom
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.oneclass as oc
import kraken.numerics.energy_splits as es

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], 1.0)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

def top_boundary(x):
    return np.isclose(x[1], 1.0)

def crack(x):
    return (x[0]<0.5)*(x[1]>(0.5-h))*(x[1]<(0.5+h))

## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



# material = Material_no_uc()
material = Params_no_uc()
material.L = 1e-3

material.g = 1e-9
material.uc = material.L
material.ψcrit = 0.0


# μ = 80.77e3
# λ = 121.15e3

# material.ν = λ/(2*(λ+μ))
# material.E = 2*μ*(1+material.ν)

# material.Gc = 2.7e3

material.E = 210e9
material.ν = 0.3
material.Gc = 2.7e3





nx =201
h = 1/nx
material.l = h*4*material.L

# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, 0]), np.array([1,1])],
#                             [nx,nx], mesh.CellType.quadrilateral)
filename = "notch.xdmf"

# msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
with io.XDMFFile(MPI.COMM_WORLD,"meshes/" + filename,"r") as infile:
    msh = infile.read_mesh()

# msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)


# c = 0.2; n=5
# msh.geometry.x[:,1] = (1-c)/L**(n-1)*msh.geometry.x[:,1]**n + c*msh.geometry.x[:,1]



# disp_bc = lambda V: [bc.get_zero_bc(V, bottom_boundary),
#                             bc.get_bc(V.sub(1), top_boundary, 1e-5)]

disp_bc  = lambda V: [bc_bottom.get_zero_bc(V, bottom_boundary),
                            bc_bottom.get_bc(V, top_boundary, default_scalar_type(np.array([0.05,0.0])))]
    

no_bc = lambda V: []
bc_d = lambda V: [bc_bottom.internal_bc(V, crack, 1.0)]
# bc_d = lambda V: [bc.internal_bc(V, lambda x: x<(x_change+0.1), 0.0)]

model = oc.viscoelastic_damage(msh, [no_bc,no_bc,no_bc], material, 
                               dt = 1)#,g = lambda d: mf.degradation_Lo2023(d,0.05))

model.p_ext = lambda u: 0.0

model.free_energy_plus = es.free_energy
# # change w
# model.damage.w = lambda d: d
# model.damage.calc_c0()
# model.damage.bounded = True

#%%
import ufl


model.setup_damage()
disps_tension = np.linspace(0.005,0.007,50)
disps_shear = np.linspace(0.0060,0.060,100)
# disps_shear = np.linspace(0.001, 0.03, 1000)

def sides_bc(x,disp):
    return np.row_stack((disp*x[1],0.0*x[1]))
disps = disps_shear

for i in range(len(disps)):
    # disp_bc  = lambda V: [bc.get_zero_bc(V, bottom_boundary),
    #                         bc.get_bc(V, top_boundary, default_scalar_type(np.array([disps[i],0.0])))]
    
    disp_bc = lambda V: [bc_bottom.get_zero_bc(V, bottom_boundary),
                         bc_bottom.get_zero_bc(V.sub(1), top_boundary),
                         bc_bottom.get_bc(V.sub(0),top_boundary, default_scalar_type(disps[i])),
                         bc_bottom.get_zero_bc(V.sub(1), left_boundary),
                            bc_bottom.get_zero_bc(V.sub(1), right_boundary)]
                        #  bc.get_bc_func(V,left_boundary, lambda x: sides_bc(x,disps[i])),
                            # bc.get_bc_func(V,right_boundary, lambda x: sides_bc(x,disps[i]))]
    model.bc_v = disp_bc(model.V)
    model.setup_elastic()

    model.fixed_point_simple(max_its=300)


    utilities.write_xdmf(path +"notchshear" + str(i) + ".xdmf", msh, \
                    [model.v,model.d],\
                    ["v","d"], t=disps[i])

