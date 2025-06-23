#%%

import numpy as np
from dolfinx import mesh, fem, plot, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
from kraken.parameters import Params_no_uc, Params_with_uc
import kraken.boundaryconditions as bc_bottom
import kraken.utilities as utilities
import kraken.mainclass as mc
import kraken.oneclass as oc
from kraken.numerics import maths_functions as mf
import kraken.numerics.energy_splits as es


d = 50
h = 100
l = 5
cell_size = l/2


dt = 1


nx = int(d/cell_size)
nz = int(h/cell_size)


def bottom(x):
    return np.isclose(x[1], 0) 
def top(x):
    return np.isclose(x[1], h)





material = Params_no_uc()
material.L = 1.0
# material.uc = 1.0
material.ρi = 0.0
material.ρw = 1.0
material.g = 1e-12
material.patm = 0.0

material.E = 30e3
material.ν = 0.2
material.ψcrit = 1.0
material.l = l
material.Gc = 1e-3


msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([d,h])],
                            [nx,nz], mesh.CellType.quadrilateral)


bc_v = lambda V: [bc_bottom.get_zero_bc(V, bottom),
                  bc_bottom.get_zero_bc(V.sub(0), top),
                    bc_bottom.get_bc(V.sub(1), top, 0.0)]

bc_d = lambda V: [bc_bottom.get_zero_bc(V, bottom),
                  bc_bottom.get_zero_bc(V, top)]                     

model = oc.viscoelastic_damage(msh, [bc_v,bc_v,bc_d], material, 1.0)

model.p_ext = lambda u: 0.0

# model.free_energy_plus = es.free_energy
# model.bounded = True
# model.w = lambda d: d


ubar = 2.5e-4


# ubar = -50e-4
if ubar>0:
    filename = "traction"
else:
    filename = "compression"


nt = 100
tt = np.linspace(0,ubar*(nt-1),nt)
F = np.zeros(nt)


facets = mesh.locate_entities_boundary(model.msh, model.msh.topology.dim - 1, top)
mesh_tags = mesh.meshtags(model.msh, model.msh.topology.dim - 1, facets, 1)
ds = ufl.Measure("ds", domain=model.msh, subdomain_data=mesh_tags)


model.setup_damage()

#%%
for i in range(nt):
    if MPI.COMM_WORLD.rank == 0:
        print("Time step: ",i)
    model.bc_v = [bc_bottom.get_zero_bc(model.V, bottom),
                        bc_bottom.get_zero_bc(model.V.sub(0), top),
                        bc_bottom.get_bc(model.V.sub(1), top, tt[i])]

    model.setup_elastic()
    model.fixed_point_simple(max_its=200)

    σzz = es.cauchy_stress(mf.ε(model.v),model.material.ν)[1,1]
    ψplus = es.free_energy_plus_spectral(mf.ε(model.v),model.material.ν)
    # F[i] = fem.assemble_scalar(fem.form(σzz*ds(1)))
    utilities.write_xdmf("outputs/" +filename + str(i) + ".xdmf",model.msh,
                        [model.v,model.d,model.Hprev,σzz,ψplus],["v","d","H","σzz","ψplus"]
                        ,t=i)


#%%
# import matplotlib.pyplot as plt

# plt.plot(tt,F)

# np.save("outputs/" + filename + ".npy",F)
# np.save("outputs/" + filename + "_t.npy",tt)
# %%
