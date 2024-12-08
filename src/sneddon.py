#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
import elasticity
import pf as pf
from material import MaterialProperties, Material_no_uc
from boundaryconditions import get_zero_bc
import utilities
import energybased as eb

L = 1
def boundary(x):
    return np.isclose(x[0], -L) | \
        np.isclose(x[0], L) | \
        np.isclose(x[1], -L) | \
        np.isclose(x[1], L)

l = 0.05




material = Material_no_uc(ρi = 0.0, ρw = 1e6, g=1.0, E = 1e9,
                                ν=0.2)
material.L = 1.0
material.l = l



msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-L, -L]), np.array([L, L])],
                            [600,600], mesh.CellType.quadrilateral)

#refine mesh

c = 0.2; n=5
msh.geometry.x[:,1] = (1-c)/L**(n-1)*msh.geometry.x[:,1]**n + c*msh.geometry.x[:,1]
c = 0.3; n=5
msh.geometry.x[:,0] = (1-c)/L**(n-1)*msh.geometry.x[:,0]**n + c*msh.geometry.x[:,0]

# material.set_l_from_mesh(msh)

bc_v = lambda V: [get_zero_bc(V, boundary, default_scalar_type)]

def crack(x,a,h):
    return (x[0]>-a)*(x[0]<a)*(x[1]>-h/2)*(x[1]<h/2)
a = 0.125
h = 0.01
π = np.pi
aeff = a + π*l/4

aeff1 = a*(1 + (π*l/4) / (a*(3*h/(8*l) + 1)))
aeff2 = a*(1 + (π*l/4) / (a*(h/(2*l) + 1)))

umax = 2*material.pwc*a*(1-material.ν**2)/material.E
umax_eff = 2*material.pwc*aeff1*(1-material.ν**2)/material.E

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
def bc_d_func(msh,V,crack):
    deactivate_cells = mesh.locate_entities(msh, msh.topology.dim, crack)
    deactivate_dofs = fem.locate_dofs_topological(V, msh.topology.dim, deactivate_cells)
    return fem.dirichletbc(default_scalar_type(1.0), deactivate_dofs, V)

# dh = phasefield.crack2phasefield(msh,l,lambda x: crack(x,a,h))

# vh = elasticity.solve(msh,bc,material,d=dh,pw=lambda u: -1.0)
crackk = lambda x: crack(x,a,h)
bc_d = lambda V: [bc_d_func(msh,V,crackk)]

vh,dh = eb.fixed_point(msh, [bc_v,bc_d], material, pw=lambda u: -1.0, max_its=2)

utilities.write_vtk("output/sneddon.pvd",msh,[vh,dh],["v","d"])

print(np.max(vh.x.array))
print(umax_eff)
# %%
