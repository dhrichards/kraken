import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type, mesh
import numpy as np
from numerics import integrators



class SurfaceSolver:
    def __init__(self, msh, bc_func, material, dt, eulerian_surfaces, k = 1e-4):
        self.msh = msh
        self.material = material
        self.dt = dt
        self.k = k
        self.eulerian_surfaces = eulerian_surfaces

        z_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        self.V = fem.functionspace(self.msh, z_el)

        self.bcs = bc_func(self.V)

        facet_indices = [], facet_markers = []
        for i in range(len(eulerian_surfaces)):
            facets = mesh.locate_entities_boundary(self.msh, self.msh.topology.dim - 1, eulerian_surfaces[i])
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, i+1))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)

        mesh_tags = mesh.meshtags(self.msh, self.msh.topology.dim - 1,
                 facet_indices[sorted_facets], facet_markers[sorted_facets])
        
        self.ds = ufl.Measure("ds", domain=self.msh, subdomain_data=mesh_tags)

    def solve(self, u):

        D = self.msh.geometry.dim

        z = ufl.TrialFunction(self.Z)
        v = ufl.TestFunction(self.Z)
        
        # ds = self.ds(1) + self.ds(2) # top and bottom boundary
        ds = sum([self.ds(i+1) for i in range(len(self.eulerian_surfaces))])
        
        acc = self.params.acc(ufl.SpatialCoordinate(self.msh)) # accumulation rate
        
        if D == 2:
            rh = integrators.RK4(
                lambda z: u[D-1] + acc - self.dt*ufl.dot(u[0], ufl.grad(z)[0]),
                self.z, self.dt)
        else:
            rh = integrators.RK4(
                lambda z: u[D-1] + acc - self.dt*ufl.dot(u[0], ufl.grad(z)[0]) \
                    - self.dt*ufl.dot(u[1], ufl.grad(z)[1]),
                self.z, self.dt)

        # Basically trying to solve the surface equation on ds, 
        # and an equation the dz/dz is constant, i.e. d2z/dz2 = 0 in the domain
        a = ufl.inner(z,v)*ds + self.k*self.dt*ufl.inner(ufl.Dx(z,D-1),ufl.Dx(v,D-1))*ufl.dx
        # +   k*self.dt*ufl.inner(ufl.grad(z), ufl.grad(v))*ufl.dx
            
    
        L = ufl.inner(rh,v)*ds

        problem = fem.petsc.LinearProblem(a, L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        return problem.solve()


