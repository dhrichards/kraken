from dolfinx import fem
import adios4dolfinx
from mpi4py import MPI
import numpy as np
import basix.ufl as bufl
from matplotlib import tri




def scalar_to_array(msh, f):
    CG1 = fem.functionspace(msh, ("CG", 1))
    # Put onto CG1 space

    # if hasattr(f,"ufl_function_space") and f.ufl_element().degree == 1:
    #             return f.x.array
    # else:
    f_CG1 = fem.Function(CG1)
    f_CG1.interpolate(fem.Expression(f, CG1.element.interpolation_points()))
    return f_CG1.x.array[:]
    

def vector_to_array(msh, f):
    fis = []
    for i in range(f.ufl_shape[0]):
        fis.append(scalar_to_array(msh, f.sub(i)))

    return fis

def dolfinx_to_array(msh, f):
    if len(f.ufl_shape) == 0:
        return scalar_to_array(msh, f)
    elif len(f.ufl_shape) == 1:
        return vector_to_array(msh, f)
    else:
        raise NotImplementedError("Only scalar and vector functions are supported.")
            

def get_triangulation(msh):
    connty = msh.topology.connectivity(2, 0)
    connty_array = np.array([connty.links(i)
            for i in range(connty.num_nodes)])
    
    return tri.Triangulation(
            msh.geometry.x[:,0], 
            msh.geometry.x[:,1], 
            triangles=connty_array)
    

def get_outline(msh):
    x,y = msh.geometry.x[:,0], msh.geometry.x[:,1]
    
    tess = get_triangulation(msh)
    
    edges = np.sort(np.vstack([
        tess.triangles[:, [0, 1]],
        tess.triangles[:, [1, 2]],
        tess.triangles[:, [2, 0]],
    ]), axis=1)

    # Count occurrences of edges
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    # Build an array of NaN-separated segments
    X = np.full((3 * len(boundary_edges),), np.nan)
    Y = np.full((3 * len(boundary_edges),), np.nan)

    for ii, e in enumerate(boundary_edges):
        X[3*ii:3*ii+2] = x[e]
        Y[3*ii:3*ii+2] = y[e]


    return X,Y


