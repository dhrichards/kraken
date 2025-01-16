import glob
import pyvista
from mpi4py import MPI
import dolfinx

from dolfinx.io import XDMFFile

if __name__ == "__main__":
    for i in range(0, 20):
        print(i)
        x = XDMFFile(MPI.COMM_WORLD, f"outputs/iceberginitial{i}.xdmf", "r")
        msh = x.read_mesh()
        topology, cell_types, x = dolfinx.plot.vtk_mesh(msh)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        plotter = pyvista.Plotter(notebook=False, off_screen=True)
        plotter.add_mesh(grid, show_edges=True)
        plotter.camera_position = "xy"
        plotter.show()
        plotter.screenshot(f"iceberg_image.{i:02}.png")
        plotter.clear()
