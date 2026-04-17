import adios4dolfinx
from .base import Damage
import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type, nls
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics.maths_functions import ε
from kraken.numerics import solvers
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np



class HigherOrder(Damage):
    def __init__(self, sim):
        super().__init__(sim)


        self.d_el_mixed = bufl.mixed_element([self.d_el, self.d_el])

        self.W = fem.functionspace(self.sim.msh, self.d_el_mixed)
        self.w = fem.Function(self.W, name="mixed function")
        self.d, self.lap = ufl.split(self.w)
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.d_prev_time, self.lap_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.w_prev_it2 = fem.Function(self.W, name="mixed function 2 iterations previous")
        self.d_prev_it, self.lap_prev_it = ufl.split(self.w_prev_it)
        self.d_prev_it2, self.lap_prev_it2 = ufl.split(self.w_prev_it2)

        self.D, _ = self.W.sub(0).collapse()

        self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        self.Hprev = fem.Function(self.H_space, name="history")


        bc_func_mod = lambda V: self.sim.bc_funcs[1](V.sub(0))
        self.bc_d = bc_func_mod(self.W)


    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        
        l0 = l/2
        c = 1-self.d

        # H = es.history_function(self.sim.momentum.ε_e, self.Hprev,
        #                     self.sim.params.ν, self.sim.params.ψcritstar,
        #                     self.sim.free_energy_plus)
        g = es.degradation_default(self.d,self.sim.params.ge_tol)
        H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        # self.F = (C3*4*l0*c*v*H + c*v - 2*l0**2*self.lap*v - l0**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
        #         -1.0*v ) * ufl.dx \
        #         - (self.lap*q + ufl.inner(ufl.grad(c), ufl.grad(q))) * ufl.dx
                 # -C3*2*(1-self.d)*l*ufl.div(self.sim.momentum.p_crack*self.sim.momentum.u)*v*ufl.dx\
        self.F = -C3*2*l*(1-self.d)*v*H*ufl.dx \
                  +(1/2)*(2*self.d*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                ) * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * ufl.dx
        

    

        C_new = 1e-2/(self.sim.params.Gc*self.sim.params.τ)

        d_dot = (self.d - self.d_prev_time)/self.sim.params.dtstar

        # self.F += C_new*l*d_dot*v*ufl.dx
        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def timestep(self):
        super().timestep()
        self.w_prev_time.x.array[:] = self.w.x.array[:]



    def solve(self):
        self.solver.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]
        assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"

    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]

    
class HigherOrderAT1(HigherOrder):
    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        

        H = es.history_function(self.sim.momentum.ε_e, self.Hprev,
                            self.sim.params.ν, self.sim.params.ψcritstar,
                            self.sim.free_energy_plus)

        HH = ufl.max_value(C3*H - 0.25*l, 0)

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        self.F = (1/(4*l))*(2*self.d*v - l**2*self.lap*v + (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)))*ufl.dx \
                -2*(1-self.d)*HH*v * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * ufl.dx
                
        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)


class PenalizedAT2(HigherOrder):
    def __init__(self, sim):
        super().__init__(sim)

        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.d_prev_time, self.lap_prev_time = ufl.split(self.w_prev_time)

    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        l0 = l/2
        c = 1-self.d

        H = ufl.max_value(self.sim.free_energy_plus(self.sim.momentum.ε_e, ν) - ψcrit, 0)
        
        γ = 1e4

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        self.F = (C3*4*l0*c*v*H + c*v - 2*l0**2*self.lap*v - l0**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                -1.0*v ) * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(c), ufl.grad(q))) * ufl.dx\
                + (γ/2)*ufl.inner(mf.negative_part(self.d - self.d_prev_time,0)**2, v)*ufl.dx
    
                
        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)


    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]


class PenalizedAT1(PenalizedAT2):

    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        # H = ufl.max_value(self.sim.free_energy_plus(self.sim.momentum.ε_e, ν) - ψcrit, 0)
        H = self.sim.free_energy_plus(self.sim.momentum.ε_e, ν)
        γ = 1e2


        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)
     
        Ψ = -2*(1-self.d)*H
        cw = 3.020
        wprime = 2 - 2*self.d
        χ2 = 1/2
        χ4 = 1/16

        # self.F = C3*Ψ*v*ufl.dx \
        #         + (1/cw)*(wprime*v/l - 2*χ2*l*self.lap*v \
        #                  -2*χ4*l**3*ufl.inner(ufl.grad(self.lap), ufl.grad(v))) * ufl.dx \
        #         - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * ufl.dx\
        #         + (γ/2)*ufl.inner(mf.negative_part(self.d - self.d_prev_time,0)**2, v)*ufl.dx


        ##working AT2
        # self.F = -C3*2*l*(1-self.d)*v*H*ufl.dx \
        #           +(1/2)*(2*self.d*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
        #         ) * ufl.dx \
        #         - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * ufl.dx\
        #         + (γ/2)*ufl.inner(mf.negative_part(self.d - self.d_prev_time,0)**2, v)*ufl.dx
    
        


        self.F = -C3*2*l*(1-self.d)*v*H*ufl.dx \
                  +(1/4)*(2*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                ) * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * ufl.dx\
                + (γ/2)*ufl.inner(mf.negative_part(self.d - self.d_prev_time,0)**2, v)*ufl.dx\
                # + (γ/2)*ufl.inner(mf.positive_part(self.d - 1,0)**2, v)*ufl.dx
    
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

        



class Bounded(HigherOrder):
    def __init__(self, sim):
        super().__init__(sim)

        self.w_lb = fem.Function(self.W, name="mixed function previous time")
        self.w_lb.sub(0).interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
        self.w_lb.sub(1).interpolate(lambda x: np.full(x.shape[1], -1e7, dtype=np.float64))
    

    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        ν = self.sim.params.ν; ψcrit = self.sim.params.ψcritstar

        l0 = l/2
        c = 1-self.d

        H = ufl.max_value(self.sim.free_energy_plus(self.sim.momentum.ε_e, ν) - ψcrit, 0)


        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        self.F = (C3*4*l0*c*v*H + c*v - 2*l0**2*self.lap*v - l0**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                -1.0*v ) * ufl.dx \
                - (self.lap*q + ufl.inner(ufl.grad(c), ufl.grad(q))) * ufl.dx
                
        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def setup_solver(self):
        super().setup_solver()
        
        self.w_ub = fem.Function(self.W)
        # d_ub, lap_ub = ufl.split(w_ub)

       
        self.w_ub.sub(0).interpolate(lambda x: np.ones(x.shape[1], dtype=np.float64))
        self.w_ub.sub(1).interpolate(lambda x: np.full(x.shape[1], 1e7, dtype=np.float64))

        # self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        # self.solver.setFunction(self.problem.F, fem.petsc.create_vector(fem.form(self.F)))
        # self.solver.setJacobian(self.problem.J, fem.petsc.create_matrix(fem.form(self.J)), P=None)

        self.solver.setType("vinewtonrsls")
        self.solver.setVariableBounds(self.w_lb.x.petsc_vec, self.w_ub.x.petsc_vec)

        # self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("cg")
        # self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("jacobi")
        # self.solver.getKSP().getPC().setFactorSolverType("mumps")

    def timestep(self):
        # Update damage variable inside mixed function, leave lower bound for laplacian unchanged
        # self.w_lb.sub(0).interpolate(self.w.sub(0))
        pass
        