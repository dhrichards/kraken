#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
import dolfinx
from dolfinx import io, mesh
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr

l = 0.05
cellfactor = 2.0
h = l/cellfactor
L = 1.0

def crack(x,a,h):
    return (x[0]>-a)*(x[0]<a)*(x[1]>=-h/2)*(x[1]<=h/2)
a = 0.125
π = np.pi

aeff = a*(1 + (π*l/4) / (a*(h/(2*l) + 1)))



refinements = 3
large_size = h*2**refinements

def boundary(x):
    return np.isclose(x[0], -L) | \
        np.isclose(x[0], L) | \
        np.isclose(x[1], -L) | \
        np.isclose(x[1], L)


def refined(x):
    return (x[1]>-2*a)*(x[1]<2*a)#*(x[0]>-2*a)*(x[0]<2*a)


nx = int(2*L/large_size)
# make sure nx is odd
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[-L, -L],
                        [L, L]],
                        [nx, nx], mesh.CellType.triangle)


msh = kr.meshes.refine_by_area(msh,refined,refinements+1)


model = kr.base.Simulation(msh)

model.tol = 5e-6


model.params.H.value = L
model.params.l.value = l

model.params.patm.value = 1.2
model.params.sea_level = -5
model.params.ρi.value = 0.0
model.params.ρc = 1.0
model.params.g.value = 1.0
model.params.E.value = 1.0
model.params.ν.value = 0.15

model.params.Kic.value = np.sqrt(1.0*model.params.E.value/(1-model.params.ν.value**2)) # for Gc = 1.0
model.params.σt.value = 0.0



u_bc = lambda V: [bc.get_zero_bc(V, boundary)]
d_bc = lambda V: [bc.internal_bc(V, lambda x: crack(x,a,h), 1.0)]

model.setup(kr.momentum.elastic.Elasticity,
            kr.damage.higherorder.AT2,
            [u_bc, d_bc])

model.damage_on = True

# model.damage.solve()
# model.momentum.solve()
model.fixed_point()
kr.utilities.write_xdmf("outputs/sneddon.xdmf",model.msh,
                        [model.momentum.u, model.damage.d,
                         model.params.patmstar],
                        ["u", "d", "patmstar"])

umax_eff = 1.0*aeff*(1-0.15**2)/model.params.E.value
v_max = MPI.COMM_WORLD.allreduce(np.max(model.momentum.u.x.array), op=MPI.MAX)
if MPI.COMM_WORLD.rank == 0:
    # print(np.max(model.d.x.array))
    print(v_max)
    print(umax_eff)