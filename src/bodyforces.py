import ufl
import phasefield as pf

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


    pw = ufl.conditional(ufl.lt(z, 0),
                         -z,
                         0.0)
    return pw


def bodyforce(ρratio,dim):
    if dim == 2:
        return ufl.Constant((0, -ρratio))
    else:
        return ufl.Constant((0, 0, -ρratio))
    


def totalforces(msh, u, d, material, pw=lambda u: water_pressure(u)):
    C1 = material.C1
    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh)

    f = bodyforce(material.ρratio, msh.geometry.dim)
    g = lambda d: pf.degradation(d)

    
    return C1 *( ufl.dot(f, u) - pw(u)*ufl.inner(ufl.grad(g(d)), u) )* ufl.dx \
        - C1 * g(d) * pw(u) *  ufl.dot(n, u) * ds
    



