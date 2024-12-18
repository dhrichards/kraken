import gmsh
import numpy as np

gmsh.initialize()
def create_iceberg_mesh(true_length, true_height, material,filename = "iceberg.msh", Hw=None):

    model = gmsh.model()

    model.add("iceberg")


    # material.L = true_height    
    nondim_length = true_length/material.L
    nondim_height = true_height/material.L

    if Hw is None:
        Hw = material.ρi/material.ρw*nondim_height

    large_size = nondim_height/5
    small_size = material.l/3
    end_size = small_size*10
    bottom_coarsening = 10.0
    crack_x = nondim_length/2 - nondim_height*0.7

    model.geo.addPoint(0, -Hw, 0, large_size, 1)
    model.geo.addPoint(crack_x, -Hw, 0, bottom_coarsening*small_size, 2)
    model.geo.addPoint(nondim_length/2, -Hw, 0, bottom_coarsening*end_size, 3)


    model.geo.addPoint(nondim_length/2, nondim_height-Hw, 0, end_size, 4)
    model.geo.addPoint(crack_x, nondim_height-Hw, 0, small_size, 5)
    model.geo.addPoint(0, nondim_height-Hw, 0, large_size, 6)


    model.geo.addLine(1, 2, 1)
    model.geo.addLine(2, 3, 2)
    model.geo.addLine(3, 4, 3)
    model.geo.addLine(4, 5, 4)
    model.geo.addLine(5, 6, 5)
    model.geo.addLine(6, 1, 6)

    model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)

    model.geo.addPlaneSurface([1], 1)

    model.geo.synchronize()

    model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 1)
    model.addPhysicalGroup(2, [1], 1)

    # write geo
    # gmsh.write("iceberg.geo_unrolled")

    model.mesh.generate(2)

    

    return model



def warp_msh(msh,L,xc,c):

    x = msh.geometry.x[:,0]

    xc = xc/L
    A = np.array([[1,1,1],\
                  [xc**3,xc**2,xc],\
                    [3*xc**2,2*xc,1]])
    
    b = np.array([1,xc,c])

    a = np.linalg.solve(A,b)

    xnew = a[0]*x**3 + a[1]*x**2 + a[2]*x

    msh.geometry.x[:,0] = xnew*L

    return a





def create_iceberg_mesh_structured(true_length, true_height, material,filename = "iceberg.msh"):

    model = gmsh.model()

    model.add("iceberg")


    # material.L = true_height    
    nondim_length = true_length/material.L
    nondim_height = true_height/material.L


    Hw = material.ρi/material.ρw*nondim_height

    large_size = nondim_height/5
    small_size = material.l/5
    end_size = small_size*10
    bottom_coarsening = 10.0
    crack_x = nondim_length/2 - nondim_height*0.7

    model.geo.addPoint(0, -Hw, 0, 0, 1)
    model.geo.addPoint(crack_x, -Hw, 0, 0, 2)
    model.geo.addPoint(nondim_length/2, -Hw, 0, 0, 3)
    model.geo.addPoint(nondim_length/2, nondim_height-Hw, 0, 4)
    model.geo.addPoint(crack_x, nondim_height-Hw, 0, 0, 5)
    model.geo.addPoint(0, nondim_height-Hw, 0, 0, 6)


    model.geo.addLine(1, 2, 1)
    model.geo.addLine(2, 3, 2)
    model.geo.addLine(3, 4, 3)
    model.geo.addLine(4, 5, 4)
    model.geo.addLine(5, 6, 5)
    model.geo.addLine(6, 1, 6)

    model.geo.addLine(2, 5, 7)

    model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)

    #Extrude


    # model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 1)
    # model.addPhysicalGroup(2, [1], 1)

    # # write geo
    # # gmsh.write("iceberg.geo_unrolled")

    
    # model.geo.mesh.setTransfiniteCurve(1, 200, "Progression", 0.9)
    # model.geo.mesh.setTransfiniteCurve(2, 100, "Progression", -0.9)
    # model.geo.mesh.setTransfiniteCurve(4, 100, "Progression", 0.9)
    # model.geo.mesh.setTransfiniteCurve(5, 200, "Progression", -0.9)

    model.geo.mesh.setTransfiniteCurve(3, 10, "Progression", 1.0)
    model.geo.mesh.setTransfiniteCurve(6, 10, "Progression", 1.0)
    model.geo.mesh.setTransfiniteCurve(7, 10, "Progression", 1.0)

    model.geo.mesh.setTransfiniteSurface(1, "Left", [1, 3, 4, 6])

    # # Recombined surface
    model.geo.mesh.setRecombine(2, 1)

    model.geo.addPlaneSurface([1], 1)

    model.geo.synchronize()

    model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 1)
    model.addPhysicalGroup(2, [1], 1)

    # write geo
    # gmsh.write("iceberg.geo_unrolled")

    model.mesh.generate(2)

    #write
    gmsh.write(filename)
    

    return model




