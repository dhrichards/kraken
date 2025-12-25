#%%
import gmsh
from dolfinx import io
from mpi4py import MPI

small_size = 0.01
Lx = 1e3/300
ρi_over_ρw = 0.9
large_size = 0.25

gmsh.initialize()
model = gmsh.model()

model.add("notched")



Hw = ρi_over_ρw

cut_x = Lx/2
eps = 1e-5


model.geo.addPoint(0, -Hw, 0, large_size, tag= 1)
model.geo.addPoint(Lx, -Hw, 0, large_size, tag= 2)


model.geo.addPoint(Lx, 1 - Hw, 0, large_size, tag =3) ## top right

model.geo.addPoint(cut_x+eps, 1 - Hw, 0, small_size, tag= 4)  ## top right cut
model.geo.addPoint(cut_x, 1-Hw -0.3, 0, small_size, tag= 5)  ## bottom right cut ## bottom left cut
model.geo.addPoint(cut_x -eps, 1 - Hw, 0, small_size, tag= 6)  ## top right cut left


model.geo.addPoint(0, 1 - Hw, 0, large_size, tag=7)

model.geo.addLine(1, 2, 1)
model.geo.addLine(2, 3, 2)
model.geo.addLine(3, 4, 3)
model.geo.addLine(4, 5, 4)
model.geo.addLine(5, 6, 5)
model.geo.addLine(6, 7, 6)
model.geo.addLine(7, 1, 7)

# model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7], 1)
model.geo.addPlaneSurface([1], 1)
model.geo.synchronize()

model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7], 1)
model.addPhysicalGroup(2, [1], 1)


# gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom

model.mesh.generate(2)

#save gmsh
# gmsh.write("icebergrefined.msh")

mesh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

filename = "notched.xdmf"

with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:

    file.write_mesh(mesh)
    # file.write_meshtags(model.mesh)

gmsh.finalize()


#%%
gmsh.initialize()
model = gmsh.model()

model.add("no_notch")
model.geo.addPoint(0, -Hw, 0, large_size, tag= 1)
model.geo.addPoint(Lx, -Hw, 0, large_size, tag= 2)
model.geo.addPoint(Lx, 1 - Hw, 0, large_size, tag =3) ## top right
model.geo.addPoint(0, 1 - Hw, 0, large_size, tag=4)

model.geo.addLine(1, 2, 1)
model.geo.addLine(2, 3, 2)
model.geo.addLine(3, 4, 3)
model.geo.addLine(4, 1, 4)

model.geo.addCurveLoop([1, 2, 3, 4], 1)
model.geo.addPlaneSurface([1], 1)
model.geo.synchronize()

model.addPhysicalGroup(1, [1, 2, 3, 4], 1)
model.addPhysicalGroup(2, [1], 1)

model.mesh.generate(2)
#save gmsh

mesh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)
filename = "no_notch.xdmf"
with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:
    file.write_mesh(mesh)
    # file.write_meshtags(model.mesh)
