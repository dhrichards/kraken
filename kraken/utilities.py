import numpy as np
from dolfinx import fem, io, plot
from mpi4py import MPI

def move_mesh(msh,uh,k=1.0):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
    uhh = fem.Function(V)
    uhh.interpolate(uh)
    msh.geometry.x[:,:msh.geometry.dim] += k*uhh.x.array.reshape((-1, msh.geometry.dim))

def mesh_sizes(mesh):
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    h = mesh.h(tdim,np.arange(num_cells))
    return h


def write_file(filename,msh,functions,names,t=0.0):
    if filename.endswith(".xdmf"):
        out_function = io.XDMFFile
    elif filename.endswith(".pvd"):
        out_function = io.VTKFile
    else:
        raise ValueError("Unknown file extension")

    for idx,f in enumerate(functions):
        # check if has function space
        if hasattr(f,"ufl_function_space"):
            if f.ufl_element().degree == 1:
                functions[idx].name = names[idx]
            else:
                # Interpolate onto order 1
                Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
                temp = fem.Function(Q)
                temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
                temp.name = names[idx]
                functions[idx] = temp

        else:
            Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
            temp = fem.Function(Q)
            temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
            temp.name = names[idx]
            functions[idx] = temp


    with out_function(MPI.COMM_WORLD, filename, "w") as file:
        file.write_mesh(msh)
        for f in functions:
            file.write_function(f,t)

def write_vtk(filename,msh,functions,names,t=0.0):

    for idx,f in enumerate(functions):
        # check if has function space
        if hasattr(f,"ufl_function_space"):
            if f.ufl_element().degree == 1:
                functions[idx].name = names[idx]
            else:
                # Interpolate onto order 1
                Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
                temp = fem.Function(Q)
                temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
                temp.name = names[idx]
                functions[idx] = temp

        else:
            Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
            temp = fem.Function(Q)
            temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
            temp.name = names[idx]
            functions[idx] = temp






    with io.VTKFile(MPI.COMM_WORLD, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(functions,t)



def write_xdmf(filename,msh,functions,names,t=0.0):

    for idx,f in enumerate(functions):
        # check if has function space
        if hasattr(f,"ufl_function_space"):
            if f.ufl_element().degree == 1:
                functions[idx].name = names[idx]
            else:
                # Interpolate onto order 1
                Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
                temp = fem.Function(Q)
                temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
                temp.name = names[idx]
                functions[idx] = temp

        else:
            Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
            temp = fem.Function(Q)
            temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
            temp.name = names[idx]
            functions[idx] = temp


    with io.XDMFFile(MPI.COMM_WORLD, filename, "w") as file:
        file.write_mesh(msh)
        for f in functions:
            file.write_function(f,t)

