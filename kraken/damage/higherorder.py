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
from kraken.boundaryconditions import get_zero_bc
from dolfinx.fem.petsc import LinearProblem



class AT2(Damage):
    '''
    Higher order AT2 phase-field damage model.
    This is solved using a mixed formulation, where the damage field and its Laplacian are solved for simultaneously.
    Irreversibility is enforced using a history variable, which is updated at each timestep.
    '''
    def __init__(self, sim):
        super().__init__(sim)


        self.d_el_mixed = bufl.mixed_element([self.d_el, self.d_el])

        self.W = fem.functionspace(self.sim.msh, self.d_el_mixed)
        self.w = fem.Function(self.W, name="mixed function")
        self.w_candidate = fem.Function(self.W)
        self.d, self.lap = ufl.split(self.w)
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.d_prev_time, self.lap_prev_time = ufl.split(self.w_prev_time)

        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")
        self.w_prev_it2 = fem.Function(self.W, name="mixed function 2 iterations previous")
        self.w_prev_it3 = fem.Function(self.W, name="mixed function 3 iterations previous")
        self.d_prev_it, self.lap_prev_it = ufl.split(self.w_prev_it)
        self.d_prev_it2, self.lap_prev_it2 = ufl.split(self.w_prev_it2)
        self.d_prev_it3, self.lap_prev_it3 = ufl.split(self.w_prev_it3)

        self.D, self.d_to_W = self.W.sub(0).collapse()
        self.d_to_W = np.asarray(self.d_to_W, dtype=np.int32)

        # self.H_space = fem.functionspace(self.sim.msh, ("DG", 1))
        # self.Hprev = fem.Function(self.H_space, name="history")


        free_boundary = lambda x: ~np.isclose(x[0], 0.0)
        self.bc_d = list(self.sim.bc_funcs[1](self.W.sub(0)))
        self.bc_d.append(
            get_zero_bc(self.W.sub(1), free_boundary)
        )


    def update_bcs(self,new_bcs):


        free_boundary = lambda x: ~np.isclose(x[0], 0.0)
        self.bc_d = list(new_bcs(self.W.sub(0)))
        self.bc_d.append(
            get_zero_bc(self.W.sub(1), free_boundary)
        )
        # self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)
        self.setup()

    def setup_solver(self):
        trial = ufl.TrialFunction(self.W)

        # Replace the current solution with a trial function.
        F_linear = ufl.replace(self.F, {self.w: trial})

        self.problem = LinearProblem(
            ufl.lhs(F_linear),
            ufl.rhs(F_linear),
            u=self.w,
            bcs=self.bc_d,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        self.solver = self.problem.solver


    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        dx = self.sim.dx
        
        
        # H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, self.Hprev)
        self.H_drive = fem.Function(
            self.H_space, name="damage driving field"
        )

        H_expr = ufl.max_value(
            self.sim.momentum.ψplus - self.sim.params.ψcritstar,
            self.Hprev,
        )

        self.H_drive_expr = fem.Expression(
            H_expr,
            self.H_space.element.interpolation_points(),
        )

        H = self.H_drive

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        # self.F = -C3*2*l*(1-self.d)*v*H*dx \
        #           +(1/2)*(2*self.d*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
        #         ) * dx \
        #         - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * dx
        self.F = (
            -2*C3*l*(1-self.d)*v*H
            + self.d*v
            + (l**2/2)*ufl.inner(ufl.grad(self.d), ufl.grad(v))
            - (l**4/16)*ufl.inner(ufl.grad(self.lap), ufl.grad(v))
            - self.lap*q
            - ufl.inner(ufl.grad(self.d), ufl.grad(q))
        ) * dx
                

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        # self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def interpolate_from_parent(self, parent):
        super().interpolate_from_parent(parent)

        self.w.sub(0).interpolate(parent.damage.w.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w.sub(1).interpolate(parent.damage.w.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        
        self.w_prev_it.sub(0).interpolate(parent.damage.w_prev_it.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it.sub(1).interpolate(parent.damage.w_prev_it.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        self.w_prev_it2.sub(0).interpolate(parent.damage.w_prev_it2.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it2.sub(1).interpolate(parent.damage.w_prev_it2.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        self.w_prev_it3.sub(0).interpolate(parent.damage.w_prev_it3.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_it3.sub(1).interpolate(parent.damage.w_prev_it3.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

        

        self.w_prev_time.sub(0).interpolate(parent.damage.w_prev_time.sub(0), cells0=self.sim.parent_cells, cells1=self.sim.cells)
        self.w_prev_time.sub(1).interpolate(parent.damage.w_prev_time.sub(1), cells0=self.sim.parent_cells, cells1=self.sim.cells)

    def timestep(self):
        super().timestep()
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()



    def solve(self):
        self.w_prev_it3.x.array[:] = self.w_prev_it2.x.array[:]
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]
        # self.solver.solve(None, self.w.x.petsc_vec)
        self.H_drive.interpolate(self.H_drive_expr)
        self.H_drive.x.scatter_forward()

        self.problem.solve()

        # self.w_candidate.x.array[:] = self.w.x.array.copy()
        # self.w.x.array[:] = self.w_prev_it.x.array[:] + 0.5*(self.w_candidate.x.array[:]-self.w_prev_it.x.array[:])
        self.w.x.scatter_forward()
        
        assert self.solver.getConvergedReason() > 0, "Nonlinear solver did not converge"

    def revert(self):
        self.w.x.array[:] = self.w_prev_time.x.array[:]



class Bounded(AT2):
    '''
    Higher order AT2 phase-field damage model.
    This is solved using a mixed formulation, where the damage field and its Laplacian are solved for simultaneously.
    Irreversibility is enforced using a bounded solver.
    '''
    def __init__(self, sim):
        super().__init__(sim)
        self.lower_bound = fem.Function(self.W, name="damage_lower_bound")
        self.upper_bound = fem.Function(self.W, name="damage_upper_bound")

    

    def setup_weak_form(self):
        C3 = self.sim.params.C3; l = self.sim.params.lstar
        dx = self.sim.momentum.dx
        
        
        H = ufl.max_value(self.sim.momentum.ψplus - self.sim.params.ψcritstar, 0.0)

        mixed_test = ufl.TestFunction(self.W)
        v, q = ufl.split(mixed_test)


        self.F = -C3*2*l*(1-self.d)*v*H*dx \
                  +(1/2)*(2*self.d*v - l**2*self.lap*v - (1/8)*l**4*ufl.inner(ufl.grad(self.lap), ufl.grad(v)) \
                ) * dx \
                - (self.lap*q + ufl.inner(ufl.grad(self.d), ufl.grad(q))) * dx
        

        self.J = ufl.derivative(self.F,self.w,ufl.TrialFunction(self.W))


        self.problem = solvers.SNESProblem(self.F, self.w, bcs=self.bc_d)

    def update_bounds(self):
        # Auxiliary Laplacian remains unrestricted.
        self.lower_bound.x.array[:] = PETSc.NINFINITY
        self.upper_bound.x.array[:] = PETSc.INFINITY

        # Only damage is bounded.
        idx = self.d_to_W
        self.lower_bound.x.array[idx] = self.w_prev_time.x.array[idx]
        self.upper_bound.x.array[idx] = 1.0

        self.lower_bound.x.scatter_forward()
        self.upper_bound.x.scatter_forward()

    def setup_solver(self):
        super().setup_solver()

        self.update_bounds()

        self.solver.setType("vinewtonrsls")
        self.solver.setVariableBounds(
            self.lower_bound.x.petsc_vec,
            self.upper_bound.x.petsc_vec,
        )

        ksp = self.solver.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")


    def solve(self):
        self.w_prev_it3.x.array[:] = self.w_prev_it2.x.array[:]
        self.w_prev_it2.x.array[:] = self.w_prev_it.x.array[:]
        self.w_prev_it.x.array[:] = self.w.x.array[:]

        self.update_bounds()

        idx = self.d_to_W
        self.w.x.array[idx] = np.clip(
            self.w.x.array[idx],
            self.lower_bound.x.array[idx],
            self.upper_bound.x.array[idx],
        )
        self.w.x.scatter_forward()

                # Separate unbounded solver: no bounds attached.
        unbounded = PETSc.SNES().create(self.sim.msh.comm)
        unbounded.setType("newtonls")

        unbounded.setFunction(
            self.problem.F,
            fem.petsc.create_vector(fem.form(self.F)),
        )
        unbounded.setJacobian(
            self.problem.J,
            fem.petsc.create_matrix(fem.form(self.J)),
        )

        unbounded.setTolerances(rtol=1.0e-9, max_it=50)

        ksp = unbounded.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        unbounded.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()

        reason = unbounded.getConvergedReason()
        unbounded.destroy()

        if reason <= 0:
            raise RuntimeError(
                f"Unbounded diagnostic solve failed: SNES reason {reason}"
            )

        # Original bounded solver, starting from the unbounded solution.
        self.solver.setType("vinewtonrsls")
        self.solver.setVariableBounds(
            self.lower_bound.x.petsc_vec,
            self.upper_bound.x.petsc_vec,
        )

        self.solver.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()

        reason = self.solver.getConvergedReason()

        PETSc.Sys.Print(
            "Bounded restart:",
            reason,
            "iterations:",
            self.solver.getIterationNumber(),
            "residual:",
            self.solver.getFunctionNorm(),
        )

        if reason <= 0:
            raise RuntimeError(
                f"Bounded damage solve failed: SNES reason {reason}"
            )


    def timestep(self):
        self.w_prev_time.x.array[:] = self.w.x.array[:]
        self.w_prev_time.x.scatter_forward()



