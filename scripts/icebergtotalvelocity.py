#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
 
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken.utilities as utilities
import kraken.total_velocity as tv
import kraken.oneclass as oc
import kraken.numerics.total_velocity_maths as tvm

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def crack(x):
    x_c = nondim_length/2 - nondim_height
    l = params.lstar
    return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return x[0]<(nondim_length/2 - 1.8*nondim_height)#*(x[1]>-60))


## check mpi size is correct
print(MPI.COMM_WORLD.size)
print(MPI.COMM_WORLD.rank)

print(MPI.Get_library_version())

true_length = 4e3
true_height = 300

hpc = False

if hpc:
    path = '/data/hpcdata/users/dancha/'
else:
    path = 'outputs/'



params = kp.Params_total_velocity()

# material = Material_with_uc()
params.L = 300.00
params.l = 30.0
params.dt = 60*60*24



nondim_length = true_length/params.L
nondim_height = true_height/params.L

Hw = params.ρi/params.ρw*nondim_height


# filename = "icebergL" + str(int(true_length/1e3)) + "l" + str(int(material.l*material.L)) + ".xdmf"

# # msh,ct,ft = io.gmshio.read_from_msh("../meshes/iceberg.msh", MPI.COMM_WORLD, rank=0, gdim=2)
# with io.XDMFFile(MPI.COMM_WORLD,"../meshes/" + filename,"r") as infile:
#     msh = infile.read_mesh()

# msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
cell_size = params.lstar/2.1


# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
#                             [int(0.5*nondim_length/cell_size),int(nondim_height/cell_size)], mesh.CellType.quadrilateral)



aspect_ratio_x = 1.0
aspect_ratio_z = 1
x_change = nondim_length/2 - 2.0*nondim_height
z_change = nondim_height - (nondim_height-Hw)*3.0

new_length = x_change/aspect_ratio_x + (nondim_length/2 - x_change)

new_height = z_change/aspect_ratio_z + (nondim_height - z_change)

nx = int(new_length/cell_size)
nz = int(new_height/cell_size)

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([new_length, new_height])],
                            [nx,nz], mesh.CellType.quadrilateral)


x = msh.geometry.x[:,0]

x[x>x_change/aspect_ratio_x] = x_change + x[x>x_change/aspect_ratio_x] - x_change/aspect_ratio_x
x[x<=x_change/aspect_ratio_x] = x[x<=x_change/aspect_ratio_x]*aspect_ratio_x

msh.geometry.x[:,0] = x

z = msh.geometry.x[:,1]
z[z>z_change/aspect_ratio_z] = z_change + z[z>z_change/aspect_ratio_z] - z_change/aspect_ratio_z
z[z<=z_change/aspect_ratio_z] = z[z<=z_change/aspect_ratio_z]*aspect_ratio_z

msh.geometry.x[:,1] = z - Hw




clamped_both = lambda V: [bc.get_zero_bc(V, left_boundary),
                            bc.get_zero_bc(V, right_boundary)]

clamped_bc = lambda V: [bc.get_zero_bc(V, left_boundary)]
symm_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V, fixed, 0.0)]
# bc_d = lambda V: [bc.internal_bc(V, lambda x: x<(x_change+0.1), 0.0)]

cliff_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                        bc.get_zero_bc(V.sub(1), bottom_boundary)]

model = tv.viscoelastic_damage(msh, [symm_bc,no_bc], params)



# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))

# model.bounded = True
# model.w = lambda d: d
# model.free_energy_plus = lambda ε,λ,μ: mf.free_energy_plus_star(ε,λ,μ,γ=1)


#%%
model.setup_velocity()
model.setup_damage()
solve_damage = False
for i in range(100):

    if i>50:
        solve_damage = True

    # model.solve_velocity()
    # model.solve_damage()
    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i)
    # model.setup_velocity()
    # model.setup_damage()
    model.fixed_point(solve_damage=solve_damage)
    
    eps_e = tvm.elastic_strain(model.σD, model.p, params.ν)
    ψplus = es.free_energy_plus_spectral(eps_e, params.ν)
    utilities.write_xdmf(path + "iceberginit" + str(i) + ".xdmf", msh,
                        [model.u, model.d,eps_e,ψplus,model.σD, model.σD_prev_time, model.p_prev_time,model.Hprev],
                        ["u", "d","epse","psi_plus","σD","σD_prev","p_prev","H"], t=i)
    model.lagrangian_update()
