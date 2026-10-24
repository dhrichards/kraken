from kraken import parameters, momentum, damage
from kraken.numerics import energy_splits as es
from kraken.numerics import maths_functions as mf
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
import numpy as np
import adios4dolfinx
import kraken as kr
from petsc4py import PETSc

class Simulation:
    '''Base class for the simulation. Contains the mesh, parameters, and solvers for momentum and damage.'''
    def __init__(self, msh):
        self.msh = msh
        self.params = parameters.Params(self.msh)
        self.order = 2
        self.quad_degree = max(4, 2 * self.order)
        self.quad_scheme =  "default"

        self.dx = ufl.Measure(
                    "dx",
                    domain=self.msh,
                    metadata={
                        "quadrature_degree": self.quad_degree,
                        "quadrature_rule": self.quad_scheme,
                    },
                )

        self.damage_on = True


        self.tol = 5e-6
        self.min_its = 2
        self.max_its = 300

        self.basal_friction = False


        b_facets = mesh.locate_entities_boundary(self.msh, self.msh.topology.dim-1, lambda x: np.isclose(x[1], 0.0))
        mesh_tags = mesh.meshtags(self.msh, self.msh.topology.dim - 1, b_facets, 1)
        ds = ufl.Measure("ds", domain=self.msh, subdomain_data=mesh_tags)
        self.ds_bottom = ds(1)


    def use_basal_friction(self,bottom_boundary,ocean_boundaries):
        self.marked_ds = kr.boundaryconditions.marked_ds(self.msh,[bottom_boundary,ocean_boundaries])
        self.basal_friction = True

    
        


    def setup(self,MomentumSolver=momentum.mixed.SemiLagrangianEpsilon,
                    DamageSolver=damage.higherorder.AT2,
                    bc_funcs=[lambda V: [], lambda V: []]):
        '''Setup the simulation with the given momentum and damage solvers, and boundary condition functions.'''


        self.bc_funcs = bc_funcs
        self.damage = DamageSolver(self)
        self.momentum = MomentumSolver(self)
        
        
        self.momentum.setup()
        self.damage.setup()
        

    def interpolate_from_parent(self, parent, parent_cells, bcs):
        '''Interpolate a state from a parent simulation (on a different mesh)
        to a child simulation (on a subset of the parent mesh).'''

        self.tol = parent.tol
        self.min_its = parent.min_its
        self.max_its = parent.max_its


        self.params.Ttop.value = parent.params.T.value
        self.params.ρi.value = parent.params.ρi.value
        self.params.ρw.value = parent.params.ρw.value
        self.params.g.value = parent.params.g.value
        self.params.E.value = parent.params.E.value
        self.params.ν.value = parent.params.ν.value
        self.params.A0.value = parent.params.A0.value
        self.params.n.value = parent.params.n.value
        self.params.H.value = parent.params.H.value
        self.params.l.value = parent.params.l.value
        self.params.dt.value = parent.params.dt.value
        self.params.patm.value = parent.params.patm.value
        self.params.ge_tol.value = parent.params.ge_tol.value
        self.params.crack_level_above_sea.value = parent.params.crack_level_above_sea.value
        self.params.viscosity_tol.value = parent.params.viscosity_tol.value

        self.params.σt.value = parent.params.σt.value
        self.params.Kic.value = parent.params.Kic.value


        self.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                                kr.damage.higherorder.AT2, bcs)
        # msh_cell_imap = self.msh.topology.index_map(self.msh.topology.dim)
        # self.cells = np.arange(msh_cell_imap.size_local + msh_cell_imap.num_ghosts)
        # self.parent_cells = entity_map.sub_topology_to_topology(self.cells, inverse=False)

        self.parent_cells = parent_cells
        self.cells = np.arange(len(self.parent_cells))
        
        self.momentum.interpolate_from_parent(parent)
        self.damage.interpolate_from_parent(parent)







        


    def timestep(self):
        '''Perform a timestep of the simulation.'''
        if self.damage_on:
            self.damage.timestep()
        self.momentum.timestep()
      

    def revert(self):
        ''' Revert the simulation to the start of the timestep in the picard iteration loop'''
        self.damage.revert()
        self.momentum.revert()

        self.setup()
     
    
    def write_checkpoint(self, filename, t, append=True):
        if append:
            adios4dolfinx.write_mesh(filename, self.msh, time = t,
                                     mode = adios4dolfinx.adios2_helpers.adios2.Mode.Append)
        else:
            adios4dolfinx.write_mesh(filename, self.msh,time = t)
            
        self.momentum.write_checkpoint(filename, t) 
        self.damage.write_checkpoint(filename, t)

    def read_checkpoint(self, filename, t=0):
        self.momentum.read_checkpoint(filename, t)
        self.damage.read_checkpoint(filename, t)

        
    def fixed_point(self, save=False, stop_bottom=False):
            '''Logic for performing fixed point iteration between
            the momentum and damage solvers.
            Returns a tuple of (flag, nits) where flag indicates the convergence status and nits is the number of iterations performed.
            #  1: converged normally
            #  2: maximum iterations reached
            #  3: bottom criterion reached before convergence
            #  4: converged and bottom criterion reached
            # -1: solver failure
                        
            Takes in as input a boolean save which if True will save the state of the simulation at each iteration to an xdmf file.
            Takes in as input a boolean stop_bottom which if True will stop the iteration if the L2 norm of the damage field on the bottom boundary exceeds 0.12.
            
            '''
            L2_old = 0.0

            one = fem.Function(self.damage.D)
            one.x.array[:] = 1.0
            area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

            area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))

            error_prev = 100

            errors = []

            self.momentum.creep_newton_weight.value = 0.0
            direct_mode = False
             
            i = 0
            while i < self.max_its:
                
                self.momentum.solve()
                if self.damage_on:
                    self.damage.solve()

        
             
               

                if save:
                    s = ufl.dev(self.momentum.σ)
                    s_e2 = ufl.inner(s, s) / 2
                    σ0 = es.cauchy_stress(self.momentum.ε_e, self.params.ν)
                    kr.plotting.write_xdmf("./outputs/iteration" + str(i) + ".xdmf",
                                self.msh, 
                                [
                                self.momentum.du,
                                self.damage.d,
                                self.damage.d_prev_it2,
                                self.damage.d_prev_it,
                                self.damage.d_prev_it3,
                                self.momentum.ψplus / self.params.ψcritstar,
                                self.momentum.ε_e,
                                s,
                                s_e2,
                                σ0,
                                self.momentum.σ,
                                self.momentum.fluidity_prev_it,
                                self.momentum.fluidity_used,
                                self.momentum.dq,

                            ],
                            [
                                "du", "d", "dprev2", "dprev", "dprev3",
                                "psi_plus", "eps_e", "s", "s_e2", "sig0", "sig",
                                "fluidity","fluidityused","dq"
                            ],
                                    t=i,quadrature_degree=self.quad_degree)
    
                

                L2_ = ufl.inner(self.damage.d,self.damage.d)*ufl.dx
                L2_rank = fem.assemble_scalar(fem.form(L2_))
                L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))



                L2_bottom_ = ufl.inner(self.damage.d,self.damage.d)*self.ds_bottom
                L2_bottom_rank = fem.assemble_scalar(fem.form(L2_bottom_))
                L2_bottom = np.sqrt(MPI.COMM_WORLD.allreduce(L2_bottom_rank, op=MPI.SUM))

                delta_d = self.damage.d - self.damage.d_prev_it

                error_rank = fem.assemble_scalar(
                    fem.form(ufl.inner(delta_d, delta_d) * self.dx)
                )
                error_L2 = np.sqrt(
                    self.msh.comm.allreduce(error_rank, op=MPI.SUM)
                ) / area
                if MPI.COMM_WORLD.rank == 0:
                    print(f"iteration {i}, error {error_L2:.3e}, L2 {L2:.3e}, L2_bottom {L2_bottom:.3e},mom_snes_its {self.momentum.solver.getIterationNumber()}, mom_snes_reason {self.momentum.solver.getConvergedReason()}")#, elastic time {self.momentum.elastic_time:.3e}, viscous time {self.momentum.viscous_time:.3e}")

                errors.append(error_L2)

                

                if self.momentum.solver.getConvergedReason() == -3: 
                    return -1,i
                
                if stop_bottom and L2_bottom > 0.12:
                    return 3, i


                i += 1
    
                converged = (
                    i >= self.min_its
                    and error_L2 < self.tol
                    and error_L2 <= error_prev
                    and error_prev < self.tol
                )

                                

                if converged:
                    # if not direct_mode:
                    #     # Retain the current displacement and local strain increments
                    #     # as the initial guess for the fully implicit solve.
                    #     self.momentum.creep_newton_weight.value = 1.0
                    #     direct_mode = True

                    #     if MPI.COMM_WORLD.rank == 0:
                    #         print("Switching from Picard to fully implicit creep")

                    #     # Require two fresh small-error iterations after switching.
                    #     error_prev = float("inf")
                    #     L2_old = L2
                    #     continue
                    if L2_bottom > 0.12:
                        return 4, i

                    return 1, i
                
                error_prev = error_L2
                L2_old = L2
            
            return 2,i


    def save_restart_state(self):
        """Save the coupled state into reusable, rank-local buffers.

        Call on every rank at a timestep boundary, before attempting
        the next physical timestep.

        Requires the same mesh and Function objects between save/restore.
        """
        if getattr(self, "_restart_state", None) is None:
            functions = []
            constants = []
            seen = set()

            # Find directly stored fields and constants once.
            # Deduplicate aliases such as momentum.w and momentum.du.
            for owner in (self, self.momentum, self.damage, self.params):
                for value in vars(owner).values():
                    if id(value) in seen:
                        continue

                    if isinstance(value, fem.Function):
                        functions.append(
                            (value, np.empty_like(value.x.array))
                        )
                        seen.add(id(value))

                    elif isinstance(value, fem.Constant):
                        constants.append(
                            (value, np.empty_like(np.asarray(value.value)))
                        )
                        seen.add(id(value))

            self._restart_state = {
                "geometry": np.empty_like(self.msh.geometry.x),
                "functions": functions,
                "constants": constants,
            }

        state = self._restart_state

        np.copyto(state["geometry"], self.msh.geometry.x)

        for function, saved in state["functions"]:
            np.copyto(saved, function.x.array)

        for constant, saved in state["constants"]:
            np.copyto(saved, np.asarray(constant.value))


    def restore_restart_state(self):
        """Restore the most recently saved coupled state on every rank.

        Does not change solver mode, advance time, or rebuild the solvers.
        """
        state = getattr(self, "_restart_state", None)
        if state is None:
            raise RuntimeError(
                "Call save_restart_state() before attempting the timestep."
            )

        np.copyto(self.msh.geometry.x, state["geometry"])

        for function, saved in state["functions"]:
            np.copyto(function.x.array, saved)

        for constant, saved in state["constants"]:
            constant.value = saved

        # The restored coefficients invalidate any constitutive cache.
        for owner in (self.momentum, self.damage):
            problem = getattr(owner, "problem", None)
            if problem is not None and hasattr(problem, "_constitutive_x"):
                problem._constitutive_x = None

    def solve_timestep(
        self,
        save=False,
        stop_bottom=False,
        trial_step = 0.2,
        max_retries=30,
    ):
        """Solve one physical timestep, halving dt after solver failure.

        Returns the existing (flag, nits) pair.
        Does not advance time or commit timestep histories.
        Preserves the final failed fields if retries are exhausted.
        """
        self.save_restart_state()
        trial_dt = float(self.params.dt.value)

        for retry in range(max_retries + 1):
            try:
                flag, nits = self.fixed_point(
                    save=save,
                    stop_bottom=stop_bottom,
                )

                return flag, nits

            except (RuntimeError, PETSc.Error) as exc:
                PETSc.Sys.Print(
                    f"Solve failed at dt={trial_dt:.6g} s: {exc}"
                )

                # Preserve the last failed fields and dt for output.
                if retry == max_retries:
                    return -1, -1

                self.restore_restart_state()

                trial_dt *= trial_step
                self.params.dt.value = trial_dt

                self.momentum.solver.destroy()
                self.momentum.setup_solver()

                PETSc.Sys.Print(
                    f"Retry {retry + 1}: dt={trial_dt:.6g} s"
                )
        


