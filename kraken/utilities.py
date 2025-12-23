import numpy as np
from dolfinx import fem, io, plot, geometry, mesh
from mpi4py import MPI
import gmsh


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
                        lstar,
                        aspect_ratios=(100,100), refine = (2.2,0.3),
                        cell_factor=2.1, refine_right=True, cell_type = mesh.CellType.triangle):
    
   

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
                                [nx,nz], cell_type)
    
    
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

    # msh.geometry.x[:,1] = z - Hw

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



def create_iceberg_gmsh_mesh(small_size, refines = [2.5, 0.5, 0.2], Lx=8e3/300, ρi_over_ρw=0.9):
    gmsh.initialize()
    model = gmsh.model()

    model.add("refined_iceberg")

    

    Hw = ρi_over_ρw

    refine_x = Lx - refines[0]

    large_size = 1/3

    cut_x = Lx - 0.1
   

    model.geo.addPoint(0, -Hw, 0, tag= 1)
    model.geo.addPoint(Lx, -Hw, 0, tag= 2)


    model.geo.addPoint(Lx, 1 - Hw, 0, tag =3) ## top right

    model.geo.addPoint(cut_x+small_size/2, 1 - Hw, 0, tag= 6)  ## top right cut
    model.geo.addPoint(cut_x+small_size/2, 1-Hw -0.1, 0, tag= 7)  ## bottom right cut
    model.geo.addPoint(cut_x -small_size/2, 1-Hw -0.1, 0, tag= 8)  ## bottom left cut
    model.geo.addPoint(cut_x -small_size/2, 1 - Hw, 0, tag= 9)  ## top right cut left


    model.geo.addPoint(refine_x, 1 - Hw, 0, tag=4)
    model.geo.addPoint(0, 1 - Hw, 0, tag=5)


    model.geo.addLine(1, 2, 1)
    model.geo.addLine(2, 3, 2)
    model.geo.addLine(3, 4, 3)
    model.geo.addLine(4, 5, 4)
    model.geo.addLine(5, 1, 5)

    model.geo.addLine(3, 6, 6)
    model.geo.addLine(6, 7, 7)
    model.geo.addLine(7, 8, 8)
    model.geo.addLine(8, 9, 9)
    model.geo.addLine(9, 4, 10)

    # model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
    model.geo.addCurveLoop([1, 2, 6,7,8,9,10,4,5], 1)
    model.geo.addPlaneSurface([1], 1)
    model.geo.synchronize()

    model.addPhysicalGroup(1, [1, 2, 3, 4, 5], 1)
    model.addPhysicalGroup(2, [1], 1)

    # refine 0.5 inward of line 2
    field = model.mesh.field
    d1 = field.add("Distance")
    field.setNumbers(d1, "EdgesList", [2])
    field.setNumber(d1, "Sampling", 100)
    t1 = field.add("Threshold")
    field.setNumber(t1, "InField", d1)
    field.setNumber(t1, "SizeMin", small_size)
    field.setNumber(t1, "SizeMax", large_size)
    field.setNumber(t1, "DistMin", refines[1])
    field.setNumber(t1, "DistMax", 1.0)

    d2 = field.add("Distance")
    field.setNumbers(d2, "EdgesList", [3])
    field.setNumber(d2, "Sampling", 100)
    t2 = field.add("Threshold")
    field.setNumber(t2, "InField", d2)
    field.setNumber(t2, "SizeMin", small_size)
    field.setNumber(t2, "SizeMax", large_size)
    field.setNumber(t2, "DistMin", refines[2])
    field.setNumber(t2, "DistMax", 0.8)

    minfield = field.add("Min")
    field.setNumbers(minfield, "FieldsList", [t1, t2])
    field.setAsBackgroundMesh(minfield)


    # field.setNumbers(1, "EdgesList", [2,3])
    # field.add("Threshold", 2)
    # field.setNumber(2, "InField", 1)
    # field.setNumber(2, "SizeMin", small_size)
    # field.setNumber(2, "SizeMax", large_size)
    # field.setNumber(2, "DistMin", 0.5)
    # field.setNumber(2, "DistMax", 1.0)
    # field.setAsBackgroundMesh(2)
    
    # gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom

    model.mesh.generate(2)

    #save gmsh
    # gmsh.write("icebergrefined.msh")

    mesh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

    filename = "icebergrefined.xdmf"

    # with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:
    #     file.write_mesh(mesh)
    #     # file.write_meshtags(model.mesh)

    gmsh.finalize()

    return mesh


