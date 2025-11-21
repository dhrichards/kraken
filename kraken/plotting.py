from dolfinx import fem
import adios4dolfinx
from mpi4py import MPI
import numpy as np
import basix.ufl as bufl



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
            

def get_connectivity(msh):
    connty = msh.topology.connectivity(2, 0)
    connty_array = np.array([connty.links(i)
            for i in range(connty.num_nodes)])
    
    return connty_array


