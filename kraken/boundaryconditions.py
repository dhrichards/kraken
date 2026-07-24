'''Utilities for creating boundary conditions in Fenicsx'''

import numpy as np
from dolfinx import fem, mesh, default_scalar_type
import ufl


def get_boundary_dofs(V,boundary):
    '''Get the degrees of freedom on a given boundary of a function space V.'''
    msh = V.mesh
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
    # boundary_dofs_x = fem.locate_dofs_topological(V, fdim, boundary_facets)


    try: # Attempt to collapse the function space
        Vcollapse, _ = V.collapse()
        spaces = (V, Vcollapse)
    except RuntimeError:
        spaces = V
    
    boundary_dofs_x = fem.locate_dofs_topological(spaces, fdim, boundary_facets)

    return boundary_dofs_x

def get_vec(V, val, dtype=default_scalar_type):
    try :
        Vcollapse, _ = V.collapse()
        vec = fem.Function(Vcollapse)
        vec.x.array[:] = val
        return vec
    except RuntimeError:
        if V.value_size == 1:
            return dtype(val)
        else:
            return dtype(np.array([val]*V.value_size))


def get_zero_vec(V,dtype=default_scalar_type):
    try :
        Vcollapse, _ = V.collapse()
        return fem.Function(Vcollapse)
    except RuntimeError:
        if V.value_size == 1:
            return dtype(0.0)
        else:
            return dtype(np.array([0]*V.value_size))
        


def get_bc(V,boundary,bc_val):
    '''Get a boundary condition on a function space V, given a boundary and a value.'''
    boundary_dofs_x = get_boundary_dofs(V,boundary)
    return fem.dirichletbc(get_vec(V,bc_val), boundary_dofs_x, V)

def internal_bc_func(V,boundary_function,function):
    '''Define an internal boundary condition on a function space V, given a boundary function (describing the boundary)
    and a function to apply on that boundary.'''
    msh = V.mesh
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
    deactivate_cells = mesh.locate_entities(msh, msh.topology.dim, boundary_function)
    deactivate_dofs = fem.locate_dofs_topological((V,function.function_space), msh.topology.dim, deactivate_cells)
    return fem.dirichletbc(function, deactivate_dofs, V)


def internal_bc(V,boundary_function,val,topology_dim=None):
    '''Define an internal boundary condition.
     Inputs are:
      V: function space V upon which the boundary condition is applied
      boundary_function: a function that returns True for points on the boundary and False for points not on the boundary
      val: the value to apply on the boundary
      topology_dim: the topology dimension of the boundary (e.g. 0 for a point bc. default is None, which will use the
        mesh dimension)'''
    if topology_dim is None:
        topology_dim = V.mesh.topology.dim

    msh = V.mesh
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
    deactivate_cells = mesh.locate_entities(msh, msh.topology.dim, boundary_function)
    deactivate_dofs = fem.locate_dofs_topological(V, msh.topology.dim, deactivate_cells)
    return fem.dirichletbc(default_scalar_type(val), deactivate_dofs, V)


def get_zero_bc(V,boundary):
    '''
    Get a zero boundary condition on a function space V, given a boundary.
    '''
    boundary_dofs_x = get_boundary_dofs(V,boundary)
    return fem.dirichletbc(get_zero_vec(V), boundary_dofs_x, V)


def get_bc_func(V,boundary,bc_expr):
    '''Define a boundary condition on a function space V, given a boundary and a function expression.'''
    boundary_dofs_x = get_boundary_dofs(V,boundary)
    try : # Attempt to collapse the function space
        Vcollapse, _ = V.collapse()
        bc_val = fem.Function(Vcollapse)
    except RuntimeError:
        bc_val = fem.Function(V)
    bc_val.interpolate(bc_expr)
    return fem.dirichletbc(bc_val, boundary_dofs_x, V)


def marked_ds(msh, boundaries):
    '''Create a measure ds with subdomain data for the given boundaries.'''

    facets = []
    for boundary in boundaries:
        boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim-1, boundary)
        facets.append(boundary_facets)

    facets = np.hstack(facets)
    values = np.hstack([np.full_like(facets[i], i+1) for i in range(len(boundaries))])
    sorted_facets = np.argsort(facets)
    mt = mesh.meshtags(msh, msh.topology.dim-1, facets[sorted_facets], values[sorted_facets])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)
    return ds



        