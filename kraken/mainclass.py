import numpy as np
from dolfinx import fem, default_real_type
from mpi4py import MPI
import ufl
import numpy as np
import basix.ufl as bufl
from kraken.models import stokes, damage, elasticity, surface


class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, material, dt, eulerian_surfaces=False):
        self.msh = msh
        self.material = material
        self.dt = dt

        h_el = bufl.element("DG", self.msh.basix_cell(), 0, dtype=default_real_type)
        self.Q_h = fem.functionspace(self.msh, h_el)

        self.Hprev = 0.0

        self.history = damage.HistorySolver(self.msh, self.material)
        self.elastic = elasticity.ElasticitySolver(self.msh, bc_funcs[0], self.material)
        self.damage = damage.DamageSolver(self.msh, bc_funcs[2], self.material)
        self.stokes = stokes.StokesSolver(self.msh, bc_funcs[1], self.material, self.dt)

        self.v = fem.Function(self.elastic.V, name="elastic displacement")
        self.d = fem.Function(self.damage.V, name="damage")
        self.u = fem.Function(self.stokes.V, name="velocity")
        self.p = fem.Function(self.stokes.Q, name="pressure")


        if eulerian_surfaces:
            self.surface = surface.SurfaceSolver(self.msh, bc_funcs[3], self.material, self.dt, eulerian_surfaces)
            self.z = fem.Function(self.surface.V, name="z")
            self.move_mesh = self.eulerian_update
        else:
            self.move_mesh = self.lagrangian_update
 

    def solve_elastic(self):
        self.elastic.solve(self.v, self.d)

    def solve_damage(self):
        self.d = self.damage.solve(self.v, self.Hprev)

    def solve_stokes(self):
        self.stokes.solve(self.u, self.p, self.d, self.v)

    
    def fixed_point(self, max_its=100, tol=1e-4, solve_stokes=False):
        L2_old = 0.0



        for i in range(max_its):

            self.solve_damage()
            self.solve_elastic()
            if solve_stokes:
                self.stokes.solve_linearised(self.u,self.p,self.d,self.v)
            # self.solver_d.solve(None, self.d.x.petsc_vec)
            # self.solve_damage_limits()
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")
            
            if error_L2 < tol:
                
                break

            L2_old = L2

        # Update history function as finished fixed point iteration
        self.Hprev = self.history.solve(self.Hprev, self.v)


    
    def lagrangian_update(self):
        V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        uhh = fem.Function(V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.dt*uhh.x.array.reshape((-1, self.msh.geometry.dim))


    def eulerian_update(self):
        self.z = self.surface.solve(self.u)

        self.msh.geometry.x[:,self.msh.geometry.dim-1] = self.z.x.array


