import ufl
import phasefield as pf
from dolfinx import default_scalar_type, fem

def water_pressure_static(msh,ρw=1.0,g=1.0):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    pw = ufl.conditional(ufl.lt(z, 0),
                         -ρw*g*z,
                         0.0)
    return pw


def water_pressure(msh,vh):
    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1] + vh[msh.geometry.dim-1]#*material.uc/material.L


    # pw = ufl.conditional(ufl.lt(z, 0),
    #                      -z,
    #                      0.0)
    pw = ufl.max_value(0.0,-z)
    return pw


def body_force(msh,ρratio):
    if msh.geometry.dim == 2:
        f = fem.Constant(msh, default_scalar_type((0, -ρratio)))
    else:
        f = fem.Constant(msh, default_scalar_type((0, 0, -ρratio)))
    return f


# def body_forces(u, d, v, f, C1, pw = None):
#     g = lambda d: pf.degradation(d)
#     return C1 *( ufl.dot(f, u) - pw(u)*ufl.inner(ufl.grad(g(d)), v) )


# def traction_forces(u, d, v, n, C1, pw):
#     g = lambda d: pf.degradation(d)
#     return C1*g(d)*pw(ufl)*ufl.inner(n,v)

# def total_forces(msh, u, d, v, material, pw = None):

#     n = ufl.FacetNormal(msh)
#     ds = ufl.Measure("ds", domain=msh)

#     if pw is None:
#         pw = lambda u: water_pressure(msh,u)

#     f = body_force(msh, material.ρratio)




#     return body_forces(msh, u, d, v, f, material.C1, pw)*ufl.dx \
#         + traction_forces(msh, u, d, v, n, material.C1, pw)*ds




