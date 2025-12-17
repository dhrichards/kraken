import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import energy_splits_deviatoric as esd
from kraken.numerics import projection_tensors as pt
from kraken.numerics import solvers
from petsc4py import PETSc
import adios4dolfinx


class Momentum:
    def __init__(self, sim):
        self.sim = sim
        

        self.V = fem.functionspace(self.sim.msh, self.sim.msh.ufl_domain().ufl_coordinate_element())


        self.DG0 = fem.functionspace(self.sim.msh, ("DG", 0))
        self.areaf = ufl.TestFunction(self.DG0)
        self.cell_area_form = fem.form(self.areaf * ufl.dx)
        self.area_0 = np.copy(fem.assemble_vector(self.cell_area_form).array)

        self.area_ratio = fem.Function(self.DG0)
        self.area_ratio.x.array[:] = 1.0
        





    def setup(self):
        self.setup_momentum()
        self.setup_solver()

    def water_pressure(self,u):
        return mf.water_pressure(self.sim.msh, u, self.sim.params.ucstar) + self.sim.params.patmstar
    
    def crack_pressure(self, u):
        # x = ufl.SpatialCoordinate(self.sim.msh)
        # return ufl.conditional(ufl.gt(x[0],25.666),1.0,0.0)*
        return mf.water_pressure(self.sim.msh, u, self.sim.params.ucstar, level=self.sim.level) + self.sim.params.patmstar
        # return mf.water_pressure_static(self.sim.msh, self.sim.level) + self.sim.params.patmstar
        
    def stress(self,ε):
        g = es.degradation_default(self.sim.damage.d,self.sim.params.ge_tol)
        σ0 = es.cauchy_stress(ε, self.sim.params.ν)
        σplus = self.sim.stress_plus(ε, self.sim.params.ν)
        # ψplus = self.sim.free_energy_plus(ε, self.sim.params.ν)
        # σpluscorrected = es.stress_plus_consistent(σplus, ψplus, self.sim.params.ψcritstar)
        σminus = σ0 - σplus
        return g*σplus+ σminus
    
    def stress_alt(self,εD,trε):
        g = es.degradation_default(self.sim.damage.d,self.sim.params.ge_tol)
        σ0 = esd.cauchy_stress(εD, trε, self.sim.params.ν)
        σplus = self.sim.stress_plus_alt(εD, trε, self.sim.params.ν)
        σminus = σ0 - σplus
        return g*σplus+ σminus
    
    def deviatoric_stress(self,εD,trε):
        g = es.degradation_default(self.sim.damage.d,self.sim.params.ge_tol)
        τ0 = 2*εD
        τplus = self.sim.deviatoric_stress_plus(εD, trε, self.sim.params.ν)
        τminus = τ0 - τplus
        return g*τplus + τminus

    
    def setup_solver(self):
        

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)


        self.solver.setTolerances(rtol=1.0e-11, max_it=150, atol=1e-13)
        self.solver.getKSP().setType("preonly")
        # self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")
 

        # self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(self.F,jit_options=dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))))
        # self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(self.J,jit_options = dict(cffi_extra_compile_args=["-std=gnu17", "-g0"]))),P=None)

        self.solver.setFunction(self.problem.F, dolfinx.fem.petsc.create_vector(fem.form(self.F)))
        self.solver.setJacobian(self.problem.J, dolfinx.fem.petsc.create_matrix(fem.form(self.J)),P=None)


    def update_bcs(self,new_bcs):
        self.bc_u = new_bcs(self.W)
        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_u)
        self.setup_solver()



    

    def solve(self):
        pass

    def timestep(self):
        pass


    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]
        # self.w_prev_it.x.array[:] = self.w_prev_time.x.array[:]


    def write_checkpoint(self, filename, t=0):
        adios4dolfinx.write_function(filename, self.w, name = "w_momentum",time = t)
        adios4dolfinx.write_function(filename, self.area_ratio, name = "area_ratio", time = t)

    def read_checkpoint(self, filename, t=0):
        adios4dolfinx.read_function(filename, self.w, name = "w_momentum", time = t)
        adios4dolfinx.read_function(filename, self.area_ratio, name = "area_ratio", time = t)


        

