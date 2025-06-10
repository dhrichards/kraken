import numpy as np
from dolfinx import fem, mesh, default_scalar_type


def get_boundary_dofs(V,boundary):
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
    boundary_dofs_x = get_boundary_dofs(V,boundary)
    return fem.dirichletbc(bc_val, boundary_dofs_x, V)


def internal_bc(V,func,val):
    msh = V.mesh
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
    deactivate_cells = mesh.locate_entities(msh, msh.topology.dim, func)
    deactivate_dofs = fem.locate_dofs_topological(V, msh.topology.dim, deactivate_cells)
    return fem.dirichletbc(default_scalar_type(val), deactivate_dofs, V)


def get_zero_bc(V,boundary):
    return get_bc(V,boundary,get_zero_vec(V))


def get_bc_func(V,boundary,bc_expr):
    boundary_dofs_x = get_boundary_dofs(V,boundary)
    bc_val = fem.Function(V)
    bc_val.interpolate(bc_expr)
    return fem.dirichletbc(bc_val, boundary_dofs_x)


        