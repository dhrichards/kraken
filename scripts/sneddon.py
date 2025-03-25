#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
from kraken.material import Material_no_uc
from kraken.boundaryconditions import get_zero_bc
import kraken.utilities as utilities
import kraken.mainclass as mc

L = 1
def boundary(x):
    return np.isclose(x[0], -L) | \
        np.isclose(x[0], L) | \
        np.isclose(x[1], -L) | \
        np.isclose(x[1], L)

l = 0.025

w_model = 'AT2'


material = Material_no_uc(ρi = 0.0, ρw = 1.0, g=1.0, E = 1.0,
                                ν=0.15, Gc=1.0)
material.L = 1.0
material.lstar = l

h = l/2
nx = int(2*L/(h/2))

msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-L, -L]), np.array([L, L])],
                            [nx,nx], mesh.CellType.quadrilateral)

#refine mesh

# c = 0.2; n=5
# msh.geometry.x[:,1] = (1-c)/L**(n-1)*msh.geometry.x[:,1]**n + c*msh.geometry.x[:,1]
# c = 0.3; n=5
# msh.geometry.x[:,0] = (1-c)/L**(n-1)*msh.geometry.x[:,0]**n + c*msh.geometry.x[:,0]

# material.set_l_from_mesh(msh)

bc_v = lambda V: [get_zero_bc(V, boundary)]

def crack(x,a,h):
    return (x[0]>-a)*(x[0]<a)*(x[1]>=-h/2)*(x[1]<=h/2)
a = 0.125
π = np.pi

if w_model == 'AT1':
    aeff = a*(1 + (π*l/4) / (a*(3*h/(8*l) + 1)))
else:
    aeff = a*(1 + (π*l/4) / (a*(h/(2*l) + 1)))

umax = 2*material.pwc*a*(1-material.ν**2)/material.E
umax_eff = 2*material.pwc*aeff*(1-material.ν**2)/material.E

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
def bc_d_func(msh,V,crack):
    deactivate_cells = mesh.locate_entities(msh, msh.topology.dim, crack)
    deactivate_dofs = fem.locate_dofs_topological(V, msh.topology.dim, deactivate_cells)
    return fem.dirichletbc(default_scalar_type(1.0), deactivate_dofs, V)

# dh = phasefield.crack2phasefield(msh,l,lambda x: crack(x,a,h))

# vh = elasticity.solve(msh,bc,material,d=dh,pw=lambda u: -1.0)
crackk = lambda x: crack(x,a,h)
bc_d = lambda V: [bc_d_func(msh,V,crackk)]



model = mc.viscoelastic_damage(msh, [bc_v,bc_v,bc_d], material, 1.0)

if w_model == 'AT1':
    model.damage.w = lambda d: d
    model.damage.calc_c0()
    model.damage.bounded = True

#overload water pressure
model.elastic.pw = lambda u: -1.0

model.fixed_point()
# model.solve_damage()
# model.solve_elastic()


utilities.write_xdmf("outputs/sneddon.xdmf",model.msh,
                        [model.v,model.d],["v","d"],t=0)


# for i in range(50):
#     if MPI.COMM_WORLD.rank == 0:
#         print(i)
#     model.solve_elastic()
#     model.solve_damage()

#     utilities.write_xdmf("outputs/sneddon" + str(i) + ".xdmf",model.msh,
#                         [model.v,model.d],["v","d"],t=i)

v_max = MPI.COMM_WORLD.allreduce(np.max(model.v.x.array), op=MPI.MAX)
if MPI.COMM_WORLD.rank == 0:
    # print(np.max(model.d.x.array))
    print(v_max)
    print(umax_eff)

# %%
