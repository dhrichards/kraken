import pyvista
from mpi4py import MPI
import dolfinx

from dolfinx.io import XDMFFile

if __name__ == "__main__":
    x = XDMFFile(MPI.COMM_WORLD, "outputs/iceberginitial19.xdmf", "r")
    msh = x.read_mesh()
    topology, cell_types, x = dolfinx.plot.vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.camera_position = "xy"
    plotter.show()
