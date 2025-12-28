#%%
import dolfinx
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import argparse
parser = argparse.ArgumentParser(description="Refine a rectangular mesh in a specific region.")
parser.add_argument("--length", type=float, default=8e3/500, help="Length of the rectangle.")
parser.add_argument("--small_size", type=float, default=0.01, help="Smallest cell size in the refined region.")
parser.add_argument("--large_size", type=float, default=0.32, help="Largest cell size in the coarse region.")
parser.add_argument("--top_fine_length", type=float, default=2.0, help="Length of the top fine region as a multiple of height.")
parser.add_argument("--full_thickness_fine_length", type=float, default=0.25, help="Length of the full thickness fine region as a multiple of height.")
parser.add_argument("--htop", type=float, default=0.15, help="Height of the top fine region as a multiple of height.")
args = parser.parse_args()

L = args.length
H = 1.0
large_size = args.large_size
small_size = args.small_size
cell_type = mesh.CellType.triangle



# make large_size a power of 2 of small_size
n_div = int(np.log2(large_size/small_size))
large_size = small_size*2**n_div
refine_steps = np.log2(large_size/small_size).astype(int)

msh = mesh.create_rectangle(MPI.COMM_WORLD,
                                [np.array([0, 0]), np.array([L, H])],
                                [int(L/large_size),int(H/large_size)], cell_type)
    
 
def cell_criterion(x):
        return (x[0] > L-args.full_thickness_fine_length*H)\
            |((x[1]>H*(1-args.htop))*(x[0]>(L-args.top_fine_length*H)))

for i in range(refine_steps+1):
    # Compute midpoints for all cells on process
    cells_local = np.arange(msh.topology.index_map(
        msh.topology.dim).size_local, dtype=np.int32)
    midpoints = mesh.compute_midpoints(
        msh, msh.topology.dim, cells_local).T

    # Check midpoint criterion and find edges connected to cells
    should_refine = np.flatnonzero(cell_criterion(midpoints)).astype(np.int32)
    msh.topology.create_entities(1)
    local_edges = mesh.compute_incident_entities(
        msh.topology, should_refine, msh.topology.dim, 1)
    msh, _, _ = mesh.refine(msh, local_edges)


#filename with length, 2 decimal places
filename = f"refined_mesh_length{L:.2f}_size{small_size:.2f}.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
    xdmf.write_mesh(msh)