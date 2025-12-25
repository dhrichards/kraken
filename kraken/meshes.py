#%%
import gmsh
from mpi4py import MPI
from dolfinx import io, mesh





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



def create_structured_unstructured_mesh(small_size, L=8e3/300, 
                                        full_thickness_fine_length=0.2, 
                                        top_fine_length=2.0, 
                                        htop=0.125,
                                        large_size=0.25):
    gmsh.initialize()
    gmsh.model.add("structured_unstructured_domain")

    # Use OpenCASCADE kernel
    gmsh.model.occ.synchronize()

    # ------------------
    # Parameters
    # ------------------
    # L  = 26.666666667
    H  = 1.0

    # full_thickness_fine_length = 0.4
    # top_fine_length = 2.0
    # htop = 0.125   # height of top structured rectangle

    xmid = L - full_thickness_fine_length     # start of right structured rectangle
    xc   = L - top_fine_length     # start of top structured rectangle

    Lc_coarse = large_size
    Lc_fine   = small_size

    # ------------------
    # Points
    # ------------------
    gmsh.model.geo.addPoint(0,     0,      0, Lc_coarse, 1)
    gmsh.model.geo.addPoint(xc,    0,      0, Lc_coarse, 2)
    gmsh.model.geo.addPoint(xmid,  0,      0, Lc_fine,   3)
    gmsh.model.geo.addPoint(L,     0,      0, Lc_fine,   4)

    gmsh.model.geo.addPoint(0,     H-htop, 0, Lc_coarse, 5)
    gmsh.model.geo.addPoint(xc,    H-htop, 0, Lc_coarse, 6)
    gmsh.model.geo.addPoint(xmid,  H-htop, 0, Lc_fine,   7)
    gmsh.model.geo.addPoint(L,     H-htop, 0, Lc_fine,   8)

    gmsh.model.geo.addPoint(0,     H,      0, Lc_coarse, 9)
    gmsh.model.geo.addPoint(xc,    H,      0, Lc_coarse, 10)
    gmsh.model.geo.addPoint(xmid,  H,      0, Lc_fine,   11)
    gmsh.model.geo.addPoint(L,     H,      0, Lc_fine,   12)

    # ------------------
    # Lines
    # ------------------
    # bottom
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)

    # verticals
    gmsh.model.geo.addLine(1, 5, 4)
    gmsh.model.geo.addLine(2, 6, 5)
    gmsh.model.geo.addLine(3, 7, 6)
    gmsh.model.geo.addLine(4, 8, 7)

    gmsh.model.geo.addLine(5, 9, 8)
    gmsh.model.geo.addLine(6, 10, 9)
    gmsh.model.geo.addLine(7, 11, 10)
    gmsh.model.geo.addLine(8, 12, 11)

    # top
    gmsh.model.geo.addLine(9, 10, 12)
    gmsh.model.geo.addLine(10, 11, 13)
    gmsh.model.geo.addLine(11, 12, 14)

    # middle horizontals
    gmsh.model.geo.addLine(5, 6, 15)
    gmsh.model.geo.addLine(6, 7, 16)
    gmsh.model.geo.addLine(7, 8, 17)

    # ------------------
    # Surfaces
    # ------------------
    gmsh.model.geo.addCurveLoop([1, 5, -15, -4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.addCurveLoop([2, 6, -16, -5], 2)
    gmsh.model.geo.addPlaneSurface([2], 2)

    gmsh.model.geo.addCurveLoop([3, 7, -17, -6], 3)
    gmsh.model.geo.addPlaneSurface([3], 3)

    gmsh.model.geo.addCurveLoop([15, 9, -12, -8], 4)
    gmsh.model.geo.addPlaneSurface([4], 4)

    gmsh.model.geo.addCurveLoop([16, 10, -13, -9], 5)
    gmsh.model.geo.addPlaneSurface([5], 5)

    gmsh.model.geo.addCurveLoop([17, 11, -14, -10], 6)
    gmsh.model.geo.addPlaneSurface([6], 6)

    # ------------------
    # Structured meshing
    # ------------------


    Nx_right = int((L - xmid)/small_size)
    Ny_full  =  int((H-htop)/small_size)
    Ny_top   = int(htop/small_size)
    Nx_mid   = int((xmid - xc)/small_size)

    # Bottom-right structured block
    gmsh.model.geo.mesh.setTransfiniteCurve(3,  Nx_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(17, Nx_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(6,  Ny_full)
    gmsh.model.geo.mesh.setTransfiniteCurve(7,  Ny_full)

    gmsh.model.geo.mesh.setTransfiniteSurface(3)
    # gmsh.model.geo.mesh.setRecombine(2, 3)

    # Top-middle structured block
    gmsh.model.geo.mesh.setTransfiniteCurve(16, Nx_mid)
    gmsh.model.geo.mesh.setTransfiniteCurve(13, Nx_mid)
    gmsh.model.geo.mesh.setTransfiniteCurve(9,  Ny_top)
    gmsh.model.geo.mesh.setTransfiniteCurve(10, Ny_top)

    gmsh.model.geo.mesh.setTransfiniteSurface(5)
    # gmsh.model.geo.mesh.setRecombine(2, 5)

    # Top-right structured block
    gmsh.model.geo.mesh.setTransfiniteCurve(17, Nx_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(14, Nx_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(10, Ny_top)
    gmsh.model.geo.mesh.setTransfiniteCurve(11, Ny_top)

    gmsh.model.geo.mesh.setTransfiniteSurface(6)
    # gmsh.model.geo.mesh.setRecombine(2, 6)

    # ------------------
    # Finalize & mesh
    # ------------------
    gmsh.model.geo.synchronize()

    # gmsh.option.setNumber("Mesh.RecombineAll", 0)
    #change all quads to tris
    # gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.TransfiniteTri", 1)

    gmsh.model.mesh.generate(2)

    # gmsh.write("mesh.msh")
    msh, ct, ft = io.gmshio.model_to_mesh(gmsh.model(), MPI.COMM_WORLD, rank=0, gdim=2)

    gmsh.finalize()

    return msh





def create_iceberg_gmsh_mesh(small_size, refines = [2.5, 0.5, 0.2], Lx=8e3/300):
    gmsh.initialize()
    model = gmsh.model()

    model.add("refined_iceberg")

    

    Hw = 0

    refine_x = Lx - refines[0]

    large_size = 1/3

    cut_x = Lx - 0.1
   

    model.geo.addPoint(0, -Hw, 0, tag= 1)
    model.geo.addPoint(Lx, -Hw, 0, tag= 2)


    model.geo.addPoint(Lx, 1 - Hw, 0, tag =3) ## top right

    # model.geo.addPoint(cut_x+small_size/2, 1 - Hw, 0, tag= 6)  ## top right cut
    # model.geo.addPoint(cut_x+small_size/2, 1-Hw -0.1, 0, tag= 7)  ## bottom right cut
    # model.geo.addPoint(cut_x -small_size/2, 1-Hw -0.1, 0, tag= 8)  ## bottom left cut
    # model.geo.addPoint(cut_x -small_size/2, 1 - Hw, 0, tag= 9)  ## top right cut left


    model.geo.addPoint(refine_x, 1 - Hw, 0, tag=4)
    model.geo.addPoint(0, 1 - Hw, 0, tag=5)


    model.geo.addLine(1, 2, 1)
    model.geo.addLine(2, 3, 2)
    model.geo.addLine(3, 4, 3)
    model.geo.addLine(4, 5, 4)
    model.geo.addLine(5, 1, 5)

    # model.geo.addLine(3, 6, 6)
    # model.geo.addLine(6, 7, 7)
    # model.geo.addLine(7, 8, 8)
    # model.geo.addLine(8, 9, 9)
    # model.geo.addLine(9, 4, 10)

    model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
    # model.geo.addCurveLoop([1, 2, 6,7,8,9,10,4,5], 1)
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
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom

    model.mesh.generate(2)

    #save gmsh
    # gmsh.write("icebergrefW
    msh, ct, ft = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)

    filename = "icebergrefined.xdmf"

    # with io.XDMFFile(MPI.COMM_WORLD,filename,"w") as file:
    #     file.write_mesh(mesh)
    #     # file.write_meshtags(model.mesh)

    gmsh.finalize()

    return msh



