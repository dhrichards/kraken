#%%
import adios4dolfinx
import kraken as kr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI

filename = '../outputs/relaxation_lo_level0.0height300Gc0.5dt1.0psicrit1.0l4.0cellfactor1.0gv_tol2.0_damagemodelAT2higher__.bp'


i = 11
msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=i)


# create simulation object to easily make correct function spaces
model = kr.base.Simulation(msh, kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.HigherOrder)

adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=i)
adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=i)

d = kr.plotting.dolfinx_to_array(msh,model.damage.d)

ψplus = model.free_energy_plus(model.momentum.ε_e, model.params.ν)

ψp = kr.plotting.dolfinx_to_array(msh, ψplus)

cty = kr.plotting.get_connectivity(msh)

tess = tri.Triangulation(
        msh.geometry.x[:,0], 
        msh.geometry.x[:,1], 
        triangles=cty)

fig,ax = plt.subplots(1,1,figsize=(6,5))
c = ax.tripcolor(tess, ψp, shading='gouraud', cmap='viridis')
# fig.colorbar(c, ax=ax, label='Damage d')

ax.set_aspect('equal')
ax.set_xlim([24.5, 26.7])
ax.set_ylim([-0.2, 0.1])