#%%
from mpi4py import MPI
from dolfinx import fem
import numpy as np
import ufl
import os
from dolfinx import io, mesh
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr
import gmsh


gmsh.initialize()

model = gmsh.model()

model.add("structured_unstructured_domain")

# Use OpenCASCADE kernel
model.occ.synchronize()

l = 0.3
size = l/5
H  = 10.0
L = 20
Lmid = 10

block_L = 4
block_H = 1

# ------------------
# Points
# ------------------
model.geo.addPoint(0,     0,      0, size, 1)
model.geo.addPoint(0,    H,      0, size, 2)
model.geo.addPoint(-Lmid + block_L, H, 0, size, 3)
model.geo.addPoint(-Lmid + block_L, H + block_H, 0, size, 4)
model.geo.addPoint(-Lmid,    H + block_H, 0, size, 5)
model.geo.addPoint(-Lmid,    H,     0, size, 6)
model.geo.addPoint(-L,     0,      0, size, 7)


model.geo.addLine(1, 2, 1)
model.geo.addLine(2, 3, 2) 
model.geo.addLine(3, 4, 3)
model.geo.addLine(4, 5, 4)
model.geo.addLine(5, 6, 5)
model.geo.addLine(6, 7, 6)
model.geo.addLine(7, 1, 7)

model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7], 1)
model.geo.addPlaneSurface([1], 1)



model.geo.synchronize()

model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7], 1)
model.addPhysicalGroup(2, [1], 1)


model.mesh.generate(2)

gmsh.write("mesh.msh")
msh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

gmsh.finalize()

def right_boundary(x):
    return np.isclose(x[0], 0)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

def top_mid(x):
    return (x[1]>H + block_H)*(x[0]>-8.1)*(x[0]<-7.9)

def block(x):
    return x[1]>H

u_bc = lambda V: [
    bc.get_zero_bc(V, bottom_boundary),
    bc.get_bc(V.sub(0), right_boundary, 0.0),
    # bc.get_bc(V.sub(1), top_mid, 0.0),
]

d_bc = lambda V: [ bc.internal_bc(V, block, 0.0) ]


model = kr.base.Simulation(msh,
                           kr.momentum.elastic.Elasticity,
                           kr.damage.lowerorder.Bounded, 
                           [u_bc, d_bc],split='dp')


model.params.H.value = 1.0
model.params.E.value = 10e6
model.params.ν.value = 0.4
model.params.l.value = l
model.params.Gc.value = 5e3
model.params.g.value = 10.0
model.params.ρi.value = 2e3
model.params.ρw.value = 0.0
model.params.ψcrit.value = 0.0
# model.params.B.value = 0.12

B = 0.12
model.params.friction_angle.value = np.arcsin(3*np.sqrt(3)*B/(2-np.sqrt(3)*B))


model.damage_on = True



model.setup()

model.momentum.solve()

model.msh.geometry.x[:,:model.msh.geometry.dim] -= model.params.ucstar_float*model.momentum.u.x.array.reshape((-1, model.msh.geometry.dim))
        
model.momentum.solve()

kr.utilities.write_xdmf("./outputs/soilslope_initial.xdmf",
                        model.msh, [model.params.ucstar*model.momentum.u, model.damage.d],
                        ["u", "d"],
                        t=0.0)
if MPI.COMM_WORLD.rank == 0:
    print(model.params.ucstar_float)
disps = np.linspace(0.01, 0.02, 10)/model.params.ucstar_float

# disps = [0.02/model.params.ucstar_float]
# factors = np.linspace(1,5,100)

for i in range(len(disps)):
    u_bc = lambda V: [
        bc.get_zero_bc(V, bottom_boundary),
        bc.get_bc(V.sub(0), right_boundary, 0.0),
        bc.get_bc(V.sub(1), top_mid, -disps[i]),
        bc.get_zero_bc(V.sub(0), top_mid)
    ]

    model.momentum.update_bcs(u_bc)
    # y = model.msh.geometry.x[:,1]
    # model.momentum.ρfactor.x.array[y>10] = factors[i]

    model.fixed_point(save=True)

    kr.utilities.write_xdmf("./outputs/soilslope_" + str(i) + ".xdmf",
                            model.msh, [model.params.ucstar*model.momentum.u, model.damage.d, model.params.B],
                            ["u", "d", "B"],
                            t=i)




