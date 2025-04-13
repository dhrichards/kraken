#%%
import numpy as np
from dolfinx import mesh, io, log, default_scalar_type, fem
from mpi4py import MPI
import numpy as np
import kraken 
from kraken.material import Material_no_uc, Material_with_uc
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken.utilities as utilities
import kraken.mainclass as mc
import kraken.oneclass as oc

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def crack(x):
    x_c = nondim_length/2 - nondim_height
    l = material.lstar
    return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return 1-((x[0]>(nondim_length/2 - 1.4*nondim_height))*(x[1]>-60))


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
material.L = 1.0
material.l = 2.0
material.Gc = 1.0
# material.set_C1_to_one()
# material.ν = 0.42
material.ψcrit = 1.0


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
cell_size = material.lstar/2.1


# msh = mesh.create_rectangle(MPI.COMM_WORLD,
#                             [np.array([0, -Hw]), np.array([nondim_length/2, nondim_height-Hw])],
#                             [int(0.5*nondim_length/cell_size),int(nondim_height/cell_size)], mesh.CellType.quadrilateral)



aspect_ratio_x = 100.0
aspect_ratio_z = 20
x_change = nondim_length/2 - 1.5*nondim_height
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

model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], material, 
                               dt = 1)#g = lambda d: mf.degradation_Lo2023(d,0.05))

# model.free_energy_plus = es.free_energy_plus_notension
# model.bounded = True
# model.w = lambda d: d
# model.free_energy_plus = lambda ε,λ,μ: mf.free_energy_plus_star(ε,λ,μ,γ=1)


#%%


# model.setup_elastic()
# model.setup_damage()

# model.solve_elastic()
# model.solve_damage()


# import ufl
# pwincrack = model.pw*ufl.inner(ufl.grad(model.g), model.v)
# pw2 = model.pw

# utilities.write_xdmf(path + "iceberginit.xdmf",msh,\
#                     [model.v,model.d,
#                       mf.principal_stress(mf.ε(model.v),material.λ,material.μ),
#                       mf.free_energy_plus(mf.ε(model.v),material.λ,material.μ),
#                       pwincrack,
#                       pw2],\
#                     ["v","d", "λ","freeenergyplus","pwincrack","pw2"],t=0)



g0 = 6.
step = 0.15
model.material.g = g0
i=0
g_end = 9.8

gs = [9.1,9.8]
for g in gs:
    model.material.g = g
    if MPI.COMM_WORLD.rank == 0:
        print(f"gravity: {model.material.g}")

    model.f = mf.body_force(model.msh, model.material.ρi, model.material.g)
    model.pw = mf.water_pressure(model.msh,model.v,model.material.ρw,model.material.g,model.material.patm)
    # have to re-setup as constants are changing
    model.setup_elastic()
    model.setup_damage()
    model.fixed_point_simple(max_its=100,tol=1e-5)

    
    utilities.write_xdmf("outputs/iceberginitialcoarse" + str(i) + ".xdmf",model.msh,\
            [model.v,model.d,
            mf.principal_stress(mf.ε(model.v),model.material.λ,model.material.μ),
                es.free_energy_plus_spectral(mf.ε(model.v),model.material.λ,model.material.μ),
                es.free_energy_plus_notension(mf.ε(model.v),model.material.λ,model.material.μ),
                es.free_energy_plus_star(mf.ε(model.v),model.material.λ,model.material.μ),
                ],\
            ["v","d","λ","spectral","notension","star"],t=i)
    i+=1


    # model.material.g += step
    # step = step*1.5
    # if model.material.g > g_end:
    #     model.material.g = g_end





# model.material.g = 9.8
# if MPI.COMM_WORLD.rank == 0:
#     print("Starting visco-elasto-damage loop")
# # model.stokes.setup_solver(model.u,model.p,model.d,model.v)
# # # model.material.g += (9.81-g0)/(steps-1)
# for i in range(300):
    
#     if MPI.COMM_WORLD.rank == 0:
#         print(str(i))
    
#     model.fixed_point(tol=1e-4)
#     # log.set_log_level(log.LogLevel.INFO)
#     model.solve_stokes()
    

#     utilities.write_xdmf(path +"iceberg" + str(i) + ".xdmf",msh,\
#                     [model.v,model.d,model.u, mf.principal_stress(mf.ε(model.v),material.ν)],\
#                     ["v","d","u", "λ"],t=i)

#     model.lagrangian_update()

