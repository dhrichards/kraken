import numpy as np
from dolfinx import fem, io, plot, geometry, mesh
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




class vtx_writer:
    def __init__(self, filename, msh, functions, names):
        self.functions = []
        for idx,f in enumerate(functions):
        # check if has function space
            if hasattr(f,"ufl_function_space"):

                if f.function_space.element.basix_element.discontinuous == False:
                    self.functions.append(f)
                    self.functions[idx].name = names[idx]
                    
                else:
                    # Interpolate onto order 1
                    Q = fem.functionspace(msh, ("CG", 1, f.ufl_shape))
                    self.functions.append(fem.Function(Q))
                    self.functions[idx].interpolate(fem.Expression(f,Q.element.interpolation_points()))
                    self.functions[idx].name = names[idx]
                   
            else:
                
                Q = fem.functionspace(msh, ("CG", 1, f.ufl_shape))
                self.functions.append(fem.Function(Q))
                self.functions[idx].interpolate(fem.Expression(f,Q.element.interpolation_points()))
                self.functions[idx].name = names[idx]
        


        self.vtx = io.VTXWriter(MPI.COMM_WORLD,
                   filename + ".bp",
                   self.functions, mesh_policy=io.VTXMeshPolicy.reuse)
        


    def write(self, functions, t=0.0):
        for idx,f in enumerate(functions):
            # check if has function space
            if hasattr(f,"ufl_function_space"):
                if f.function_space.element.basix_element.discontinuous == False:
                    self.functions[idx] = f
                else:
                    Q = self.functions[idx].function_space
                    self.functions[idx].interpolate(fem.Expression(f,Q.element.interpolation_points()))

            else:
                Q = self.functions[idx].function_space
                self.functions[idx].interpolate(fem.Expression(f,Q.element.interpolation_points()))

        self.vtx.write(t)
    



def write_xdmf(filename,msh,functions,names,t=0.0):

    for idx,f in enumerate(functions):
        # check if has function space
        if hasattr(f,"ufl_function_space"):
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



def extract_line(points,msh,functions):

    # Interpolate expression onto order 1 function space
    for idx,f in enumerate(functions):
        # check if has function space
        if hasattr(f,"ufl_function_space"):
            if f.ufl_element().degree == 1:
                pass
                # functions[idx].name = names[idx]
            else:
                # Interpolate onto order 1
                Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
                temp = fem.Function(Q)
                temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
                # temp.name = names[idx]
                functions[idx] = temp

        else:
            Q = fem.functionspace(msh, ("Lagrange", 1, f.ufl_shape))
            temp = fem.Function(Q)
            temp.interpolate(fem.Expression(f,Q.element.interpolation_points()))
            # temp.name = names[idx]

            functions[idx] = temp
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)


    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])


    points_on_proc = np.array(points_on_proc, dtype=np.float64)


    

    func_vals = []
    for f in functions:
        func_vals.append(f.eval(points_on_proc, cells))

    return points_on_proc, func_vals






def create_refined_mesh(nondim_length, nondim_height, 
                        lstar,Hw,
                        aspect_ratios=(100,100), refine = (2.2,0.3),
                        cell_factor=2.1, refine_right=True ):
    
   

    cell_size = lstar/cell_factor

    aspect_ratio_x = aspect_ratios[0]
    aspect_ratio_z = aspect_ratios[1]



    x_change = nondim_length/2 - refine[0]
    z_change = nondim_height - refine[1]

    if refine_right:
        new_length = x_change/aspect_ratio_x + (nondim_length/2 - x_change)
    else:
        new_length = x_change + (nondim_length/2 - x_change)/aspect_ratio_x

    new_height = z_change/aspect_ratio_z + (nondim_height - z_change)

    nx = int(new_length/cell_size)
    nz = int(new_height/cell_size)

    msh = mesh.create_rectangle(MPI.COMM_WORLD,
                                [np.array([0, 0]), np.array([new_length, new_height])],
                                [nx,nz], mesh.CellType.quadrilateral)
    
    
    x = msh.geometry.x[:,0]

    if refine_right:
        x[x>x_change/aspect_ratio_x] = x_change + x[x>x_change/aspect_ratio_x] - x_change/aspect_ratio_x
        x[x<=x_change/aspect_ratio_x] = x[x<=x_change/aspect_ratio_x]*aspect_ratio_x
    else:
        x[x>x_change] = x_change + (x[x>x_change] - x_change)*aspect_ratio_x



    msh.geometry.x[:,0] = x

    z = msh.geometry.x[:,1]
    z[z>z_change/aspect_ratio_z] = z_change + z[z>z_change/aspect_ratio_z] - z_change/aspect_ratio_z
    z[z<=z_change/aspect_ratio_z] = z[z<=z_change/aspect_ratio_z]*aspect_ratio_z

    msh.geometry.x[:,1] = z - Hw

    return msh



# import pyvista
# def plot_damage_state(u, d):
#     """
#     Plot the displacement and damage field with pyvista
#     """

#     mesh = u.function_space.mesh



#     topology, cell_types, geometry = plot.vtk_mesh(mesh)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
#     plotter.view_xy()
#     plotter.add_axes()
#     plotter.set_scale(5,5)

#     plotter = pyvista.Plotter(
#         title="Damage state", window_size=[800, 300], shape=(1, 2)
#     )

#     topology, cell_types, x = plot.vtk_mesh(mesh)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    
#     plotter.subplot(0, 0)
#     plotter.add_text("Displacement", font_size=11)
#     vals = np.zeros((x.shape[0], 3))
#     vals[:,:len(u)] = u.x.array.reshape((x.shape[0], len(u)))
#     grid["u"] = vals
#     warped = grid.warp_by_vector("u", factor=0.1)
#     actor_1 = plotter.add_mesh(warped, show_edges=False)
#     plotter.view_xy()

#     plotter.subplot(0, 1)

#     plotter.add_text("Damage", font_size=11)

#     grid.point_data["alpha"] = d.x.array
#     grid.set_active_scalars("alpha")
#     plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, clim=[0, 1])
#     plotter.view_xy()
#     if not pyvista.OFF_SCREEN:
#        plotter.show()


