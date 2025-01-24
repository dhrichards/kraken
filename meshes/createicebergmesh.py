#%%
import sys
sys.path.append("src")
import gmsh
from dolfinx import io
from kraken.material import Material_no_uc
from mpi4py import MPI


true_length = 4e3
true_height = 300

material = Material_no_uc()
material.L = true_height
material.τ = 3600*24  


material.l = 3.0/material.L

gmsh.initialize()
model = gmsh.model()

model.add("iceberg")


# material.L = true_height    
nondim_length = true_length/material.L
nondim_height = true_height/material.L

Hw = material.ρi/material.ρw*nondim_height

large_size = nondim_height/5
small_size = material.l/5
end_size = small_size*8
bottom_coarsening = 10.0
crack_x = nondim_length/2 - nondim_height*1.1

model.geo.addPoint(0, -Hw, 0, large_size, 1)
model.geo.addPoint(crack_x, -Hw, 0, bottom_coarsening*small_size, 2)
model.geo.addPoint(nondim_length/2, -Hw, 0, bottom_coarsening*end_size, 3)


model.geo.addPoint(nondim_length/2, nondim_height-Hw, 0, end_size, 4)
model.geo.addPoint(crack_x, nondim_height-Hw, 0, small_size, 5)
model.geo.addPoint(0, nondim_height-Hw, 0, large_size, 6)


model.geo.addLine(1, 2, 1)
model.geo.addLine(2, 3, 2)
model.geo.addLine(3, 4, 3)
model.geo.addLine(4, 5, 4)
model.geo.addLine(5, 6, 5)
model.geo.addLine(6, 1, 6)

model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)

model.geo.addPlaneSurface([1], 1)

model.geo.synchronize()

model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 1)
model.addPhysicalGroup(2, [1], 1)

# write geo
# gmsh.write("iceberg.geo_unrolled")

model.mesh.generate(2)





# gmsh.write("iceberg.msh")

mesh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

filename = "icebergL" + str(int(true_length/1e3)) + "l" + str(int(material.l*material.L)) + ".xdmf"

with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:
    file.write_mesh(mesh)
    # file.write_meshtags(model.mesh)