#%%
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import ufl.mathfunctions
import kraken.utilities as utilities
import numpy as np  
import kraken.boundaryconditions as bc
import kraken.mainclass as mc
from kraken.material import Material_no_uc



L = 1000e3 # length of 1 wavelength
H = 100
Lmax = 4



nz = 15
nx = 100

s0 = 5e-1
s = 1e-5
R = 200e3


def accum(x):
    return np.maximum(0.0, np.minimum(s0, s*(R-np.abs(x))))

def accumulation(x):
    absx = ufl.max_value(0.0,x[0]) + ufl.max_value(0.0,-x[0])
    return ufl.max_value(0.0,ufl.min_value(s0, s*(R-absx)))

def bottom_boundary(x):
    return np.isclose(x[1],0.0)

def left_boundary(x):
    return np.isclose(x[0],-L//2)

def right_boundary(x):
    return np.isclose(x[0],L//2)

def top_boundary(x):
    return np.isclose(x[1],H)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([-L//2, 0]), np.array([L//2, H])],
                            [nx,nz], mesh.CellType.triangle)


facets = mesh.locate_entities_boundary(msh, 1, left_boundary)
bc_u = lambda V: [bc.get_zero_bc(V, bottom_boundary),
                    bc.get_zero_bc(V, left_boundary),
                    bc.get_zero_bc(V, right_boundary)]
bc_f = lambda V: []
bc_z = lambda V: [bc.get_zero_bc(V, bottom_boundary)]
    # bc.get_bc_func(V, left_boundary, lambda x: x[1]),
    #                 bc.get_bc_func(V, right_boundary, lambda x: x[1]),
    #                 ]
material = Material_no_uc()
material.τ = 1.0
material.L = 1.0
material.A = 1e-16

dt = 4

model = mc.viscoelastic_damage(msh, [bc_f,bc_u,bc_f,bc_z], material,
                                dt,
                               [top_boundary],accumulation)



# model.solve_stokes()
# model.move_mesh()

x = ufl.SpatialCoordinate(msh)

utilities.write_xdmf("outputs/eismint.xdmf", msh, [model.u,model.z, accumulation(x)], ["u","z","a"])

# from matplotlib import pyplot as plt

# x = np.linspace(0,200e3,1000)
# plt.plot(x, accum(x))
# # grid
# plt.grid()
z_original = fem.Function(model.surface.V)
z_original.interpolate(lambda x: x[msh.geometry.dim-1])

#%%
for i in range(2000):
    if MPI.COMM_WORLD.rank == 0:
        print(i)
    model.solve_stokes()
    # model.solve_fabric_complex()
    
    # a2 = ot.a2calc(model.f)

    model.msh.geometry.x[:,model.msh.geometry.dim-1] = z_original.x.array.real
    utilities.write_xdmf("outputs/EISMINT" + str(i) + ".xdmf", 
                         msh, [model.u, model.z], ["u", "z"], t=i)
    model.msh.geometry.x[:,model.msh.geometry.dim-1] = model.z.x.array.real

    model.move_mesh()


    
#%%

# msh = model.msh

# Q = fem.functionspace(msh, ("Lagrange", 1, model.u.ufl_shape))
# u = fem.Function(Q)
# u.interpolate(fem.Expression(model.u,Q.element.interpolation_points()))
# x = msh.geometry.x
# vals = np.zeros((x.shape[0], 3))
# vals[:,:len(u)] = u.x.array.reshape((x.shape[0], len(u)))

# from matplotlib import pyplot as plt

# # unique x vals
# xvals = np.unique(x[:,0])

# # get largest y val for each unique x value
# yvals = np.zeros_like(xvals)
# uvals = np.zeros_like(xvals)

# for i, xval in enumerate(xvals):
#     yvals[i] = np.max(x[x[:,0] == xval,1])

# # get idx of largest y val for each unique x value
# for i, xval in enumerate(xvals):
#     idx = np.argmax(x[x[:,0] == xval,1])
#     uvals[i] = vals[x[:,0] == xval][idx,0]




# plt.plot(xvals, yvals)
# plt.figure()
# plt.plot(xvals, uvals)
    

# #%%

# # plot over y at fixed x

# xslice = xvals[50]

# y = x[x[:,0] == xslice,1]
# u = vals[x[:,0] == xslice,0]

# u_analytic = -Uin*(y/H)**2 + Uin
# plt.plot(y,u)
# plt.plot(y,u_analytic)
