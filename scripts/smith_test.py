#%%
import numpy as np
import ufl
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.material import Material_no_uc, Material_with_uc
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.mainclass as mc

def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)



## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 2e3
true_height = 300

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



# material = Material_no_uc()
material = Material_with_uc()
material.l = 20.0
# material.Gc = 20.0
material.uc = material.L

# material.L = true_height
# material.τ = 3600*24
nondim_length = true_length/material.L
nondim_height = true_height/material.L

Hw = material.ρi/material.ρw*nondim_height


# filename = "icebergL" + str(int(true_length/1e3)) + "l" + str(int(material.l*material.L)) + ".xdmf"

# # msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
# with io.XDMFFile(MPI.COMM_WORLD,"../meshes/" + filename,"r") as infile:
#     msh = infile.read_mesh()

# msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
cell_size = material.lstar/3.2


nx = int((nondim_length/2)/cell_size)
nz = int(nondim_height/cell_size)

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
                            [nx,nz], mesh.CellType.quadrilateral)




v_bc = lambda V: [bc.get_zero_bc(V.sub(1), bottom_boundary),
                  bc.get_zero_bc(V.sub(0), left_boundary),
                    bc.get_bc(V.sub(0), right_boundary, 1e-3/nondim_length)]
no_bc = lambda V: []

model = mc.viscoelastic_damage(msh, [v_bc,no_bc,no_bc], material, 
                               dt = 1)#,g = lambda d: mf.degradation_Lo2023(d,0.05))


# # change w
# model.damage.w = lambda d: d
# model.damage.calc_c0()
# model.damage.bounded = True


# Change water pressure to constant tensile stress
σt0 = material.ρratio*(1-material.ρratio)/2
side_facets = mesh.locate_entities_boundary(msh, msh.topology.dim-1, right_boundary)
mesh_tags = mesh.meshtags(model.msh, msh.topology.dim-1, side_facets, 1)
ds = ufl.Measure("ds", domain=model.msh, subdomain_data=mesh_tags)
model.elastic.ds = ds(1)
model.elastic.pw = lambda u: 0.0

disp = np.logspace(-3,0,50)/nondim_length

#%%

# model.elastic.pw = lambda u: -0*σt0
# model.elastic.solve(model.v,model.d,model.u)

# model.fixed_point_simple()

# utilities.write_xdmf(path +"smith_elasticonly" + ".xdmf", msh, \
#                     [model.v,model.d],\
#                     ["v","d"], t=0)





for i in range(len(disp)):
    if MPI.COMM_WORLD.rank == 0:
        print(i)
    disp_bc = lambda V: [bc.get_zero_bc(V.sub(1), bottom_boundary),
                  bc.get_zero_bc(V.sub(0), left_boundary),
                    bc.get_bc(V.sub(0), right_boundary, disp[i])]
    model.elastic.bcs = disp_bc(model.elastic.V)
    model.fixed_point_simple(max_its=300)
    
    utilities.write_xdmf(path +"smith" + str(i) + ".xdmf", msh, \
                    [model.v,model.d],\
                    ["v","d"], t=i)

