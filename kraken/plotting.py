"""
Plotting utilities for Kraken.

Functions in this module enable writing to xdmfw as well as provide conversion between DOLFINx meshes and
Matplotlib-compatible representations for visualisation and post-processing.
"""
from dolfinx import fem, io
import adios4dolfinx
from mpi4py import MPI
import numpy as np
import basix.ufl as bufl
from matplotlib import tri
import basix
import ufl
from ufl.algorithms import extract_coefficients


def write_xdmf(
    filename, msh, functions, names, t=0.0, *,
    quadrature_degree=None,
):
    """
    Write functions or UFL expressions to XDMF.

    Ordinary fields are interpolated to CG1.
    Fields involving quadrature coefficients are averaged into DG0.

    quadrature_degree must match the quadrature fields' degree.
    This assumes they use the default Basix quadrature rule.
    """
    if len(functions) != len(names):
        raise ValueError("functions and names must have the same length.")

    def is_quadrature_element(element):
        if getattr(element, "family_name", "").lower() == "quadrature":
            return True

        return any(
            is_quadrature_element(e)
            for e in element.sub_elements
        )

    def uses_quadrature(expression):
        return any(
            is_quadrature_element(c.ufl_element())
            for c in extract_coefficients(expression)
        )

    expressions = [ufl.as_ufl(f) for f in functions]
    quadrature_flags = [uses_quadrature(f) for f in expressions]

    if any(quadrature_flags):
        if quadrature_degree is None:
            raise ValueError(
                "Provide quadrature_degree for quadrature-based fields."
            )

        points, weights = basix.make_quadrature(
            msh.basix_cell(), quadrature_degree
        )

        cell_map = msh.topology.index_map(msh.topology.dim)
        cells = np.arange(cell_map.size_local, dtype=np.int32)

        # Physical integration weights, including the geometry Jacobian.
        if cells.size:
            jacobian = fem.Expression(
                abs(ufl.JacobianDeterminant(msh)), points
            )
            detJ = jacobian.eval(msh, cells).reshape(
                cells.size, len(weights)
            )
            physical_weights = detJ * weights[None, :]
            cell_volumes = physical_weights.sum(axis=1)

    with io.XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)

        for f, name, use_quadrature in zip(
            expressions, names, quadrature_flags
        ):
            if not use_quadrature:
                # Ordinary finite-element fields: continuous output.
                V = fem.functionspace(
                    msh, ("Lagrange", 1, f.ufl_shape)
                )
                output = fem.Function(V, name=name)

                output.interpolate(
                    fem.Expression(
                        f, V.element.interpolation_points()
                    )
                )

            else:
                # Quadrature-based fields: one average per cell.
                V = fem.functionspace(
                    msh, ("DG", 0, f.ufl_shape)
                )
                output = fem.Function(V, name=name)

                # Construct collectively, including ranks with no cells.
                expression = fem.Expression(f, points)

                if cells.size:
                    ncomp = int(np.prod(f.ufl_shape))

                    values = expression.eval(msh, cells).reshape(
                        cells.size, len(weights), ncomp
                    )

                    averages = np.einsum(
                        "cq,cqk->ck", physical_weights, values
                    ) / cell_volumes[:, None]

                    cell_dofs = np.array(
                        [V.dofmap.cell_dofs(c)[0] for c in cells],
                        dtype=np.int32,
                    )

                    output.x.array.reshape(-1, ncomp)[
                        cell_dofs
                    ] = averages

            output.x.scatter_forward()
            file.write_function(output, t)



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
    """
    Convert a scalar or vector DOLFINx/UFL function to NumPy arrays.
    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        The mesh on which the function is defined.
    f : ufl.core.expr.Expr or dolfinx.fem.Function
        Scalar or vector-valued UFL expression or DOLFINx function.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Nodal values of the function. Scalar fields return a single array;
        vector fields return a list containing one array per component.

    Raises
    ------
    NotImplementedError
        If the input function has more than one tensor dimension.
    """
    if len(f.ufl_shape) == 0:
        return scalar_to_array(msh, f)
    elif len(f.ufl_shape) == 1:
        return vector_to_array(msh, f)
    else:
        raise NotImplementedError("Only scalar and vector functions are supported.")
            

def get_triangulation(msh):
    """
    Extract the boundary outline of a two-dimensional triangular mesh.

    Boundary edges are identified as edges belonging to only one triangle.
    The output is formatted as NaN-separated line segments, making it
    directly compatible with Matplotlib ``plot`` calls.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        A two-dimensional triangular DOLFINx mesh.

    Returns
    -------
    X : numpy.ndarray
        x-coordinates of boundary line segments with NaN separators.
    Y : numpy.ndarray
        y-coordinates of boundary line segments with NaN separators.
    """
    connty = msh.topology.connectivity(2, 0)
    connty_array = np.array([connty.links(i)
            for i in range(connty.num_nodes)])
    
    return tri.Triangulation(
            msh.geometry.x[:,0], 
            msh.geometry.x[:,1], 
            triangles=connty_array)
    

def get_outline(msh):
    """
    Extract the boundary outline of a two-dimensional triangular mesh.

    Boundary edges are identified as edges belonging to only one triangle.
    The output is formatted as NaN-separated line segments, making it
    directly compatible with Matplotlib ``plot`` calls.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        A two-dimensional triangular DOLFINx mesh.

    Returns
    -------
    X : numpy.ndarray
        x-coordinates of boundary line segments with NaN separators.
    Y : numpy.ndarray
        y-coordinates of boundary line segments with NaN separators.
    """
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


