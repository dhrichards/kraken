from kraken import parameters
from dolfinx import fem
import ufl
from mpi4py import MPI
import numpy as np

class Simulation:
    def __init__(self, msh, bc_funcs, MomentumSolver, DamageSolver):
        self.msh = msh
        self.params = parameters.Params_with_uc(self.msh)
        self.bc_funcs = bc_funcs



        self.momentum = MomentumSolver(self)
        self.damage = DamageSolver(self)


    def setup(self, **kwargs):
        self.momentum.setup()
        self.damage.setup(**kwargs)


        
    def fixed_point(self, max_its=100, tol=1e-4, min_its=2, solve_damage=True):
            L2_old = 0.0

            one = fem.Function(self.damage.D)
            one.x.array[:] = 1.0
            area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

            area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))

            error_prev = 100
            
            for i in range(max_its):
                
                if solve_damage:
                    self.damage.solve()
                self.momentum.solve()
    
                

                L2_ = ufl.inner(self.damage.d,self.damage.d)*ufl.dx
                L2_rank = fem.assemble_scalar(fem.form(L2_))
                L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

                error_L2 = np.abs(L2 - L2_old)/area
                if MPI.COMM_WORLD.rank == 0:
                    print(f"iteration {i}, error {error_L2}")

                if i>min_its-1:
                    if (error_L2 < tol) and (error_prev < tol):
                        break
                
                error_prev = error_L2
                L2_old = L2





