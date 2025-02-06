import numpy as np
from dolfinx import fem, default_real_type
from mpi4py import MPI
import ufl
import numpy as np
import basix.ufl as bufl
from kraken.models import stokes, damage, elasticity, surface
from kraken import utilities


class viscoelastic_damage:
    def __init__(self, msh, bc_funcs, material, dt, eulerian_surfaces=False,acc=lambda x: 0.0):
        self.msh = msh
        self.material = material
        self.dt = dt

        self.Hprev = 0.0

        self.history = damage.HistorySolver(self.msh, self.material)
        self.elastic = elasticity.ElasticitySolver(self.msh, bc_funcs[0], self.material, self.dt)
        self.stokes = stokes.StokesSolver(self.msh, bc_funcs[1], self.material, self.dt)
        self.damage = damage.DamageSolver(self.msh, bc_funcs[2], self.material)
       
        self.v = fem.Function(self.elastic.V, name="elastic displacement")
        self.d = fem.Function(self.damage.V, name="damage")
        self.u = fem.Function(self.stokes.V, name="velocity")
        self.p = fem.Function(self.stokes.Q, name="pressure")


        # self.elastic.setup(self.v, self.d, self.u)
        self.stokes.setup(self.u, self.p, self.d, self.v)



        if eulerian_surfaces:
            self.surface = surface.SurfaceSolver(self.msh, bc_funcs[3], self.material, self.dt, eulerian_surfaces, acc)
            self.z = fem.Function(self.surface.V, name="z")
            self.z.interpolate(lambda x: x[self.msh.geometry.dim-1])
            self.move_mesh = self.eulerian_update
        else:
            self.move_mesh = self.lagrangian_update
 

    def solve_elastic(self):
        # self.v = self.elastic.solve_linearised(self.v, self.d)
        self.elastic.solve(self.v, self.d, self.u)

    def solve_damage(self):
        self.damage.solve(self.v, self.Hprev, self.d)
        # self.d = self.damage.solve(self.v, self.Hprev, self.d)

    def solve_stokes(self):
        # self.stokes.solve(self.u, self.p, self.d, self.v)
        self.stokes.solve()

    
    def fixed_point(self, max_its=100, tol=1.5e-4, solve_stokes=False):
        L2_old = 0.0

        self.converged = False
        d_old = self.d.copy()
        v_old = self.v.copy()

        for i in range(max_its):

            self.solve_elastic()
            self.solve_damage()
            if solve_stokes:
                # self.stokes.solve_linearised(self.u,self.p,self.d,self.v)
                self.stokes.solve()
            # self.solver_d.solve(None, self.d.x.petsc_vec)
            # self.solve_damage_limits()
            

            L2_ = ufl.inner(self.d,self.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")

            if error_L2 > 1.0: # this is a test for the whole region being damaged
                break
            
            if i>0:
                if error_L2 < tol:
                    self.converged = True
                    break

            L2_old = L2

        # Update history function as finished fixed point iteration
        if self.converged:
            self.Hprev = self.history.solve(self.Hprev, self.v)
        else:
            self.d = d_old
            self.v = v_old

        

    def gravity_loop(self, g0=6.6, step=0.3, save=False, solve_stokes=False):

        self.material.g = g0

        done_final = False
    
        i=0
        g_end = 9.7
        
        # while self.material.g<9.8:
        while done_final == False:
            if MPI.COMM_WORLD.rank == 0:
                print(f"gravity: {self.material.g}")
            
            
            self.fixed_point(solve_stokes=solve_stokes)

            if self.converged:
                if save:
                    utilities.write_xdmf("outputs/iceberginitial" + str(i) + ".xdmf",self.msh,\
                            [self.v,self.d,self.u],\
                            ["v","d","u"],t=i)
                i+=1


                if self.material.g == g_end:
                    done_final = True
                else:
                    self.material.g += step
                    step = step*1.5
                    if self.material.g > g_end:
                        self.material.g = g_end

                

            else:
                step = step/2.0
                self.material.g -= step


                



        




  
         


    
    def lagrangian_update(self):
        V = fem.functionspace(self.msh, ("Lagrange", 1, (self.msh.geometry.dim, )))
        uhh = fem.Function(V)
        uhh.interpolate(self.u)
        self.msh.geometry.x[:,:self.msh.geometry.dim] += self.dt*uhh.x.array.reshape((-1, self.msh.geometry.dim))


    def eulerian_update(self):
        self.z = self.surface.solve_nitshe(self.u,self.z)

        self.msh.geometry.x[:,self.msh.geometry.dim-1] = self.z.x.array


