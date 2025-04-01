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
from kraken.numerics import maths_functions as mf

L = 2.0
a_aim = 0.1
h_aim = a_aim/10
nx = int(L/h_aim)

h = L/nx
l = 3*h
# get a as a multiple of grid spacing
a = int(a_aim/h)*h



w = h # crack width


w_model = 'AT2'


def crack(x):
    return (x[0]>=-1.001*a)*(x[0]<=1.001*a)*(x[1]>-1e-6)*(x[1]<1.001*w)

def boundary(x):
    return np.isclose(x[0], -L/2) | \
        np.isclose(x[0], L/2) | \
        np.isclose(x[1], -L/2) | \
        np.isclose(x[1], L/2)

def bottom(x):
    return np.isclose(x[1], 0) & (x[0] > 1.001*a) 
def top(x):
    return np.isclose(x[1], L)

def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

def crack_boundary(x):
    return np.isclose(x[1], 0) & (x[0] < 1.001*a)




material = Material_with_uc()

material.L = 1.0
material.uc = 1.0
material.ρi = 0.0
material.ρw = 1.0
material.g = 1.0
material.ψcrit = 0.0
material.l = l


material.E = 1
material.ν = 0.2
material.Gc = 1
pw0 = 1e-3

# msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, L])],
#                             [nx,nx], mesh.CellType.quadrilateral)
msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-L/2, -L/2]), np.array([L/2, L/2])],
                            [nx,nx], mesh.CellType.quadrilateral)
#refine mesh

# c = 0.2; n=5
# msh.geometry.x[:,1] = (1-c)/L**(n-1)*msh.geometry.x[:,1]**n + c*msh.geometry.x[:,1]
# c = 0.3; n=5
# msh.geometry.x[:,0] = (1-c)/L**(n-1)*msh.geometry.x[:,0]**n + c*msh.geometry.x[:,0]

# material.set_l_from_mesh(msh)

bc_v = lambda V: [bc.get_zero_bc(V, boundary)]

# bc_v = lambda V: [  
#                 bc.get_zero_bc(V.sub(1), bottom),
#                 bc.get_zero_bc(V, top), 
#                 bc.get_zero_bc(V.sub(0), left), 
#                 bc.get_zero_bc(V, right)]

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
bc_d = lambda V: [bc.internal_bc(V, crack, 1.0)]

# bc_d = lambda V: [bc.get_bc(V, crack_boundary, 1.0)]


π = np.pi

if w_model == 'AT1':
    aeff_yoi = a*(1 + (π*l/4) / (a*(3*h/(8*l) + 1)))
else:
    aeff_yoi = a*(1 + (π*l/4) / (a*(h/(2*l) + 1)))

aeff_jakub = a + π*l/4

def umax(a):
    return 2*pw0*a*(1-material.ν**2)/material.E




g = lambda d: mf.degradation_default(d)
model = mc.viscoelastic_damage(msh, [bc_v,bc_v,bc_d], material, 1.0, g=g)

if w_model == 'AT1':
    model.damage.w = lambda d: d
    model.damage.calc_c0()
    model.damage.bounded = True

# According to Jakob
pwc = np.sqrt(material.Gc*material.E/(np.pi*a*(1-material.ν**2)))

# #%%
# pws = np.linspace(1.0,2.0,50)

# for i in range(50):
#     model.elastic.pw = lambda u: pws[i]
#     model.fixed_point_simple(max_its=200)
#     utilities.write_xdmf("outputs/sneddon" + str(i) + ".xdmf",model.msh,
#                         [model.v,model.d],["v","d"],t=pws[i])

#%%
model.elastic.pw = lambda u: pw0

# model.solve_damage()
# model.solve_elastic()
model.fixed_point_simple(max_its=10, tol=-1)

utilities.write_xdmf("outputs/sneddon.xdmf",model.msh,
                    [model.v,model.d],["v","d"])


print("saved")
v_max = MPI.COMM_WORLD.allreduce(np.max(model.v.x.array), op=MPI.MAX)
if MPI.COMM_WORLD.rank == 0:
    # print(np.max(model.d.x.array))
    print("Vmax: ", v_max)
    print("Vmax Jakub: ", umax(aeff_jakub))
    print("Vmax YOI: ", umax(aeff_yoi))




# %%
