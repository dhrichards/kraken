from kraken import parameters, utilities
from kraken.numerics import energy_splits as es
from kraken.numerics import energy_splits_deviatoric as esd
from kraken.numerics import maths_functions as mf
from kraken.numerics import projection_tensors as pt
from dolfinx import fem
import ufl
from mpi4py import MPI
import numpy as np
import adios4dolfinx

class Simulation:
    def __init__(self, msh, MomentumSolver, DamageSolver, 
                 bc_funcs=[lambda V: [], lambda V: []],
                   level=0.0, split="lo"):
        self.msh = msh
        self.params = parameters.Params_with_uc(self.msh)
        self.bc_funcs = bc_funcs
        self.level = level

        if split == "lo":
            self.free_energy_plus = es.free_energy_plus_lo
            self.stress_plus = es.stress_plus_lo
            self.stress_plus_alt = esd.stress_plus_lo
            self.deviatoric_stress_plus = esd.deviatoric_stress_plus_lo
            self.free_energy_plus_alt = esd.free_energy_plus_lo
        elif split == "spectral":
            self.free_energy_plus = es.free_energy_plus_spectral
            self.stress_plus = es.stress_plus_spectral
            self.stress_plus_alt = esd.stress_plus_spectral
            self.deviatoric_stress_plus = esd.deviatoric_stress_plus_spectral
            self.free_energy_plus_alt = esd.free_energy_plus_spectral
        elif split == "dp":
            self.free_energy_plus = es.free_energy_plus_dp
            self.stress_plus = es.stress_plus_dp
            self.stress_plus_alt = esd.stress_plus_dp
            self.deviatoric_stress_plus = esd.deviatoric_stress_plus_dp
            self.free_energy_plus_alt = esd.free_energy_plus_dp
        elif split == "star":
            self.free_energy_plus = es.free_energy_plus_star
            self.stress_plus = es.stress_plus_star
        elif split == "amor":
            self.free_energy_plus = es.free_energy_plus_amor
            self.stress_plus = es.stress_plus_amor
        elif split == "none":
            self.free_energy_plus = es.free_energy
            self.stress_plus = es.cauchy_stress
            self.stress_plus_alt = esd.cauchy_stress
            self.deviatoric_stress_plus = lambda εD,trε,ν: 2*εD
            self.free_energy_plus_alt = esd.free_energy
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

        self.setup()
     
    
    def write_checkpoint(self, filename, t=0):
        if t == 0:
            adios4dolfinx.write_mesh(filename, self.msh,time = t)

            dictofparams = { 'rhoi' : self.params.ρi.value,
                             'rhow' : self.params.ρw.value,
                             'g' : self.params.g.value,
                             'E' : self.params.E.value,
                             'nu' : self.params.ν.value,
                             'A' : self.params.A0.value,
                             'n' : self.params.n.value,
                             'Gc' : self.params.Gc.value,
                             'L' : self.params.L.value,
                             'l' : self.params.l.value,
                             'sigmacrit' : self.params.σcrit.value,
                             'psicrit' : self.params.ψcrit.value,
                                'dt' : self.params.dt.value,
                                'patm': self.params.patm.value,
                                'gv_tol': self.params.gv_tol.value,
                                'T': self.params.T.value
                             }
            adios4dolfinx.write_attributes(filename, MPI.COMM_WORLD, 'params', dictofparams)

        else:
            adios4dolfinx.write_mesh(filename, self.msh, time = t,
                                     mode = adios4dolfinx.adios2_helpers.adios2.Mode.Append)
            
        self.momentum.write_checkpoint(filename, t) 
        self.damage.write_checkpoint(filename, t)

    def read_checkpoint(self, filename, t=0):
        dictofparams = adios4dolfinx.read_attributes(filename, MPI.COMM_WORLD, 'params')
        self.params.ρi.value = dictofparams['rhoi']
        self.params.ρw.value = dictofparams['rhow']
        self.params.g.value = dictofparams['g']
        self.params.E.value = dictofparams['E']
        self.params.ν.value = dictofparams['nu']
        self.params.A0.value = dictofparams['A']
        self.params.n.value = dictofparams['n']
        self.params.Gc.value = dictofparams['Gc']
        self.params.L.value = dictofparams['L']
        self.params.l.value = dictofparams['l']
        self.params.σcrit.value = dictofparams['sigmacrit']
        self.params.ψcrit.value = dictofparams['psicrit']
        self.params.dt.value = dictofparams['dt']
        self.params.patm.value = dictofparams['patm']
        self.params.gv_tol.value = dictofparams['gv_tol']

        self.momentum.read_checkpoint(filename, t)
        self.damage.read_checkpoint(filename, t)

        
    def fixed_point(self, max_its=100, tol=1e-4, min_its=2, solve_damage=True, save=False):
            L2_old = 0.0

            one = fem.Function(self.damage.D)
            one.x.array[:] = 1.0
            area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

            area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))

            error_prev = 100

            errors = []
             
            i = 0
            while i < max_its:
                
                if solve_damage:
                    self.damage.solve()
                self.momentum.solve()
                # if solve_mass:
                #     self.mass.solve()

                if save:
                    η = mf.viscosity(mf.εD(self.momentum.vel_prev_it), self.params.n, 1.e-15, 1.0)
                    utilities.write_xdmf("./outputs/iteration" + str(i) + ".xdmf",
                                self.msh, [self.momentum.u,self.damage.d,
                                        self.momentum.u_e, self.momentum.u_v, η
                                        ],
                                        ["u","d",
                                        "ue","uv","viscosity"
                                        ],
                                    t=i)
    
                

                L2_ = ufl.inner(self.damage.d,self.damage.d)*ufl.dx
                L2_rank = fem.assemble_scalar(fem.form(L2_))
                L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

                error_L2 = np.abs(L2 - L2_old)/area
                if MPI.COMM_WORLD.rank == 0:
                    print(f"iteration {i}, error {error_L2}, mom_snes_its {self.momentum.solver.getIterationNumber()}, mom_snes_reason {self.momentum.solver.getConvergedReason()}")

                errors.append(error_L2)

                if self.momentum.solver.getConvergedReason() == -3: #and self.params.gv_tol.value < 0.999e-3:
                    # return with a flag to tell the caller to do something
                    return -1

                    # self.params.gv_tol.value *= 10
                    # self.revert()
                    # i = 0 
                    # if MPI.COMM_WORLD.rank == 0:
                    #     print("Reverting and setting gv_tol to ", self.params.gv_tol.value)
                    # continue
                # else:
                i += 1
    

                

                if i >=min_its and (error_L2 < tol) and (error_L2 <= error_prev) and (error_prev < tol):
                    break
                
                error_prev = error_L2
                L2_old = L2
            
            return 1





