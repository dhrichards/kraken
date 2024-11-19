import copy



def dimensionalise_mesh(mesh, material):
    """Dimensionalise mesh"""
    mesh = copy.copy(mesh)
    mesh.geometry.x[:,:mesh.geometry.dim] *= material.L
    return mesh

def nondimensionalise_mesh(mesh, material):
    """Nondimensionalise mesh"""
    mesh = copy.copy(mesh)
    mesh.geometry.x[:,:mesh.geometry.dim] /= material.L
    return mesh

def dimensionalise_displacement(u, material):
    """Dimensionalise displacement"""
    return u*material.uc

def nondimensionalise_displacement(u, material):
    """Nondimensionalise displacement"""
    return u/material.uc


