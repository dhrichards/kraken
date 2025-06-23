#%%
import gmsh
from dolfinx import io
from mpi4py import MPI

gmsh.initialize()
gmsh.model.add("experiment")

dim = 2

scale = 1e-3
radius = 52*scale
notch_width = 1*scale
notch_depth = 26*scale
thickness = 42*scale

size_small = 0.5*scale
size_large = 5*scale

z_el = int(thickness/size_small)



p1 = gmsh.model.geo.addPoint(0, 0, 0, size_large)
p2 = gmsh.model.geo.addPoint(radius, 0, 0, size_large)
p2m = gmsh.model.geo.addPoint(-radius, 0, 0, size_large)
p3 = gmsh.model.geo.addPoint(0, radius, 0, size_small)
p4 = gmsh.model.geo.addPoint(0, notch_depth, 0, size_small)
p5 = gmsh.model.geo.addPoint(notch_width/2, notch_depth, 0, size_small)
p5m = gmsh.model.geo.addPoint(-notch_width/2, notch_depth, 0, size_small)
p6 = gmsh.model.geo.addPoint(notch_width/2, 0, 0, size_large)
p6m = gmsh.model.geo.addPoint(-notch_width/2, 0, 0, size_large)



l1 = gmsh.model.geo.addLine(p6, p2)
l1m = gmsh.model.geo.addLine(p2m, p6m)
l2 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
l2m = gmsh.model.geo.add_circle_arc(p3, p1, p2m)
# l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p5m, p5)
l5 = gmsh.model.geo.addLine(p5, p6)
l5m = gmsh.model.geo.addLine(p6m, p5m)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l2m, l1m, l5m, l4, l5])

surface = gmsh.model.geo.addPlaneSurface([loop])

if dim == 3:
    gmsh.model.geo.extrude([(2, surface)], 0, 0, thickness, numElements=[z_el])


gmsh.model.geo.synchronize()

#save geo
# gmsh.write("notchtest.geo_unrolled")

phys1 = gmsh.model.addPhysicalGroup(1, [l1,l2,l2m,l1m,l5m,l4,l5])
phys2 = gmsh.model.addPhysicalGroup(2, [surface])

if dim == 3:
    phys3 = gmsh.model.addPhysicalGroup(3, [1])

# model.mesh.generate(2)


gmsh.model.mesh.generate(dim)
gmsh.write("notchtest.msh")

model = gmsh.model()
mesh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=dim)

filename = "experiment" + str(dim) + "d.xdmf"

with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:
    file.write_mesh(mesh)

gmsh.finalize()