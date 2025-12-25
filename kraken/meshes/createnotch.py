#%%
import gmsh
from dolfinx import io
from kraken.parameters import Params_no_uc
from mpi4py import MPI


size = 1/201
notch_thick = 5e-3

gmsh.initialize()
model = gmsh.model()

model.add("notchtest")


model.geo.addPoint(0, 0, 0, size, 1)
model.geo.addPoint(1, 0, 0, size, 2)
model.geo.addPoint(1, 1, 0, size, 3)
model.geo.addPoint(0, 1, 0, size, 4)
model.geo.addPoint(0, 0.5+notch_thick/2, 0, size, 5)
model.geo.addPoint(0.5, 0.5, 0, size, 6)
model.geo.addPoint(0, 0.5-notch_thick/2, 0, size, 7)

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

#save geo
# gmsh.write("notchtest.geo_unrolled")

model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7], 1)
model.addPhysicalGroup(2, [1], 1)

model.mesh.generate(2)

# gmsh.write("notchtest.msh")

mesh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

filename = "notch.xdmf"

with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:
    file.write_mesh(mesh)
