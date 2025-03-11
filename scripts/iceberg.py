#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.material import Material_no_uc
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.utilities as utilities
import kraken.mainclass as mc

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def crack(x):
    x_c = nondim_length/2 - nondim_height
    l = material.l
    return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return x[0]<(nondim_length/2 - 1.4*nondim_height)


## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 16e3
true_height = 300

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



material = Material_no_uc()
material.L = true_height
material.τ = 3600*24
nondim_length = true_length/material.L
nondim_height = true_height/material.L

material.l = 6.0/material.L


Hw = material.ρi/material.ρw*nondim_height


# filename = "icebergL" + str(int(true_length/1e3)) + "l" + str(int(material.l*material.L)) + ".xdmf"

# # msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
# with io.XDMFFile(MPI.COMM_WORLD,"../meshes/" + filename,"r") as infile:
#     msh = infile.read_mesh()

# msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
cell_size = material.l/3.2



aspect_ratio = 100.0
x_change = nondim_length/2 - 1.5*nondim_height

new_length = x_change/aspect_ratio + (nondim_length/2 - x_change)

nx = int(new_length/cell_size)
nz = int(nondim_height/cell_size)

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, -Hw]), np.array([new_length, nondim_height-Hw])],
                            [nx,nz], mesh.CellType.quadrilateral)


x = msh.geometry.x[:,0]

x[x>x_change/aspect_ratio] = x_change + x[x>x_change/aspect_ratio] - x_change/aspect_ratio
x[x<=x_change/aspect_ratio] = x[x<=x_change/aspect_ratio]*aspect_ratio

msh.geometry.x[:,0] = x


#%%
clamped_both = lambda V: [bc.get_zero_bc(V, left_boundary),
                            bc.get_zero_bc(V, right_boundary)]

clamped_bc = lambda V: [bc.get_zero_bc(V, left_boundary)]
symm_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]
# bc_d = lambda V: [bc.internal_bc(V, lambda x: x<(x_change+0.1), 0.0)]

cliff_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1), bottom_boundary)]

model = mc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], material, 
                               dt = 1)
                            #    g = lambda d: mf.degradation_Lo2023(d,2))


# # change w
# model.damage.w = lambda d: d
# model.damage.calc_c0()
# model.damage.bounded = True

#%%
import ufl
# gs = [0.1, 6.6, 6.7, 6.9, 7.0, 7.02, 7.3, 9.0,9.2,9.3,9.5]
# gs = [9.5]
# gs = np.linspace(6.9,9.8,40)
# i =0
# for g in gs:
#     model.material.g = g
#     if MPI.COMM_WORLD.rank == 0:
#         print(model.material.g)
   
#     model.fixed_point(tol=1e-4,solve_stokes=False,max_its=100)


#     ψ = mf.free_energy_plus(mf.ε(model.v),material.ν)
#     ψP = mf.free_energy_plus_P(mf.ε(model.v),mf.ε(model.v),material.ν)
#     utilities.write_xdmf(path + "iceberginitial" + str(i) + ".xdmf",msh,\
#                     [model.v,model.d,ψ,ψP],\
#                     ["v","d","eps","epsP"],t=i)
#     i+=1
model.gravity_loop(save=True)
model.material.g = 9.8
if MPI.COMM_WORLD.rank == 0:
    print("Starting visco-elasto-damage loop")
# model.stokes.setup_solver(model.u,model.p,model.d,model.v)
# # model.material.g += (9.81-g0)/(steps-1)
for i in range(300):
    
    if MPI.COMM_WORLD.rank == 0:
        print(str(i))
    
    model.fixed_point(tol=1e-4,solve_stokes=True)
    # log.set_log_level(log.LogLevel.INFO)
    # model.solve_stokes()
    

    utilities.write_xdmf(path +"iceberg" + str(i) + ".xdmf",msh,\
                    [model.v,model.d,model.u, mf.principal_stress(mf.ε(model.v),material.ν)],\
                    ["v","d","u", "λ"],t=i)

    model.lagrangian_update()

