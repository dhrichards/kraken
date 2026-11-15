"""Quadratic Bernstein elements and the free-boundary Hessian C0IP form.

Bernstein coefficients in [0, 1] bound the polynomial throughout each cell.
This avoids the between-node overshoot of quadratic Lagrange elements.
"""
import copy

import basix
import basix.ufl
import numpy as np
import ufl


def quadratic_bernstein(cell):
    """Continuous P2 Bernstein basis on an interval, triangle or tetrahedron."""
    if cell not in (basix.CellType.interval, basix.CellType.triangle,
                    basix.CellType.tetrahedron):
        raise ValueError("Hageman Hessian damage currently requires simplex cells")
    reference = basix.create_element(
        basix.ElementFamily.P, cell, 2, basix.LagrangeVariant.equispaced,
    )
    x, M = copy.deepcopy(reference.x), copy.deepcopy(reference.M)
    geometry = basix.geometry(cell)
    # b_ij = 2*p(midpoint_ij) - (p(vertex_i) + p(vertex_j))/2.
    # Vertex coefficients are the vertex values. These are dual to the
    # non-negative basis lambda_i^2, 2*lambda_i*lambda_j.
    for edge, vertices in enumerate(basix.topology(cell)[1]):
        ends = geometry[vertices]
        x[1][edge] = np.vstack((ends, np.mean(ends, axis=0)))
        M[1][edge] = np.array([-.5, -.5, 2.]).reshape(1, 1, 3, 1)
    element = basix.create_custom_element(
        cell, [], reference.wcoeffs, x, M, 0,
        basix.MapType.identity, basix.SobolevSpace.H1, False, 2, 2,
        basix.PolysetType.standard,
    )
    return basix.ufl.wrap_element(element)


def hessian_form(u, v, msh, dx, penalty, coefficient=1):
    """Symmetric C0 interior penalty approximation of integral Hess(u):Hess(v).

    Only INTERIOR facets receive consistency and penalty terms. On external
    facets, the Hessian energy supplies the free moment/shear/corner conditions;
    no artificial Laplacian or normal-gradient boundary condition is imposed.
    The optional positive coefficient is included in cell, moment and penalty terms.
    """
    H_u, H_v = ufl.grad(ufl.grad(u)), ufl.grad(ufl.grad(v))
    n = ufl.FacetNormal(msh)
    # Cell altitude relative to a facet, robust to anisotropic triangles.
    h = msh.topology.dim * ufl.CellVolume(msh) / ufl.FacetArea(msh)
    b = ufl.as_ufl(coefficient)
    inv_h = ufl.max_value(b('+') / h('+'), b('-') / h('-'))
    dS = ufl.Measure('dS', domain=msh, metadata=dx.metadata())
    moment_u = ufl.avg(b*ufl.dot(n, ufl.dot(H_u, n)))
    moment_v = ufl.avg(b*ufl.dot(n, ufl.dot(H_v, n)))
    jump_u, jump_v = ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)
    return (b*ufl.inner(H_u, H_v) * dx
            - (moment_u * jump_v + moment_v * jump_u) * dS
            + penalty * inv_h * jump_u * jump_v * dS)