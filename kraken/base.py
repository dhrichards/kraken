from kraken import parameters
from kraken.numerics import energy_splits as es
from dolfinx import fem
import ufl
from mpi4py import MPI
import numpy as np
import adios4dolfinx

class Simulation:
    def __init__(self, msh, bc_funcs, MomentumSolver, DamageSolver, level=0.0, split="lo"):
        self.msh = msh
        self.params = parameters.Params_with_uc(self.msh)
        self.bc_funcs = bc_funcs
        self.level = level
        self.T = -20.0  # Default temperature

        if split == "lo":
            self.free_energy_plus = es.free_energy_plus_lo
            self.stress_plus = es.stress_plus_lo
        elif split == "spectral":
            self.free_energy_plus = es.free_energy_plus_spectral
            self.stress_plus = es.stress_plus_spectral
        elif split == "dp":
            self.free_energy_plus = es.free_energy_plus_dp
            self.stress_plus = es.stress_plus_dp
        elif split == "star":
            self.free_energy_plus = es.free_energy_plus_star
            self.stress_plus = es.stress_plus_star
        elif split == "amor":
            self.free_energy_plus = es.free_energy_plus_amor
            self.stress_plus = es.stress_plus_amor
        elif split == "none":
            self.free_energy_plus = es.free_energy
            self.stress_plus = es.cauchy_stress
        elif split == "lo_3d":
            self.free_energy_plus = es.free_energy_plus_lo_3d
            self.stress_plus = es.stress_plus_lo_3d
        else:
            raise ValueError(f"Unknown energy split: {split}")



        self.momentum = MomentumSolver(self)
        self.damage = DamageSolver(self)
        # self.mass = Mass(self)



    def setup(self):
        
        self.momentum.setup()
        self.damage.setup()
        # self.mass.setup()
        


    def timestep(self):
        # self.mass.solve()
        # self.mass.timestep()
        self.damage.timestep()
        self.momentum.timestep()
        # self.mass.timestep()

    def revert(self):
        self.damage.revert()
        self.momentum.revert()
     


        
    def fixed_point(self, max_its=100, tol=1e-4, min_its=2, solve_damage=True, solve_mass=True):
            L2_old = 0.0

            one = fem.Function(self.damage.D)
            one.x.array[:] = 1.0
            area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

            area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))

            error_prev = 100

            errors = []
            
            for i in range(max_its):
                
                if solve_damage:
                    self.damage.solve()
                self.momentum.solve()
                # if solve_mass:
                #     self.mass.solve()
    
                

                L2_ = ufl.inner(self.damage.d,self.damage.d)*ufl.dx
                L2_rank = fem.assemble_scalar(fem.form(L2_))
                L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

                error_L2 = np.abs(L2 - L2_old)/area
                if MPI.COMM_WORLD.rank == 0:
                    print(f"iteration {i}, error {error_L2}")

                errors.append(error_L2)

                if i>min_its-1:
                    if (error_L2 < tol) and (error_L2 <= error_prev) and (error_prev < tol):
                        break
                
                error_prev = error_L2
                L2_old = L2
            
            return errors





