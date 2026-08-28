from kraken import parameters, momentum, damage
from kraken.numerics import energy_splits as es
from kraken.numerics import maths_functions as mf
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
import numpy as np
import adios4dolfinx
import kraken as kr
class Simulation:
    '''Base class for the simulation. Contains the mesh, parameters, and solvers for momentum and damage.'''
    def __init__(self, msh):
        self.msh = msh
        self.params = parameters.Params(self.msh)
        

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
     
    
    def write_checkpoint(self, filename, t=0):
        if t == 0:
            adios4dolfinx.write_mesh(filename, self.msh,time = t)
        else:
            adios4dolfinx.write_mesh(filename, self.msh, time = t,
                                     mode = adios4dolfinx.adios2_helpers.adios2.Mode.Append)
            
        self.momentum.write_checkpoint(filename, t) 
        self.damage.write_checkpoint(filename, t)

    def read_checkpoint(self, filename, t=0):
        self.momentum.read_checkpoint(filename, t)
        self.damage.read_checkpoint(filename, t)

        
    def fixed_point(self, save=False, stop_bottom=False):
            '''Logic for performing fixed point iteration between
            the momentum and damage solvers.
            Returns a tuple of (flag, nits) where flag indicates the convergence status and nits is the number of iterations performed.
            flag = 1: converged
            flag = 2: max iterations reached
            flag = -1: error occurred (e.g., solver did not converge)
            
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
             
            i = 0
            while i < self.max_its:
                
                self.momentum.solve()
                if self.damage_on:
                    self.damage.solve()
                
             
               

                if save:
                    kr.plotting.write_xdmf("./outputs/iteration" + str(i) + ".xdmf",
                                self.msh, [self.momentum.u,self.damage.d,
                                           self.momentum.ψplus/self.params.ψcritstar,
                                        #    self.momentum.p_crack(self.momentum.du),
                                        self.momentum.u_e, self.momentum.du_v
                                        ],
                                        ["u","d",
                                            "psi_plus",
                                            # "p_c",
                                        "ue","uv",
                                        ],
                                    t=i)
    
                

                L2_ = ufl.inner(self.damage.d,self.damage.d)*ufl.dx
                L2_rank = fem.assemble_scalar(fem.form(L2_))
                L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))



                L2_bottom_ = ufl.inner(self.damage.d,self.damage.d)*self.ds_bottom
                L2_bottom_rank = fem.assemble_scalar(fem.form(L2_bottom_))
                L2_bottom = np.sqrt(MPI.COMM_WORLD.allreduce(L2_bottom_rank, op=MPI.SUM))

                error_L2 = np.abs(L2 - L2_old)/area
                if MPI.COMM_WORLD.rank == 0:
                    print(f"iteration {i}, error {error_L2:.3e}, L2 {L2:.3e}, L2_bottom {L2_bottom:.3e},mom_snes_its {self.momentum.solver.getIterationNumber()}, mom_snes_reason {self.momentum.solver.getConvergedReason()}, elastic time {self.momentum.elastic_time:.3e}, viscous time {self.momentum.viscous_time:.3e}")

                errors.append(error_L2)

                

                if self.momentum.solver.getConvergedReason() == -3: 
                    return -1,i
                
                if stop_bottom and L2_bottom > 0.12:
                    return -1,i

                i += 1
    

                

                if i >=self.min_its and (error_L2 < self.tol) and (error_L2 <= error_prev) and (error_prev < self.tol):
                    return 1,i
                
                error_prev = error_L2
                L2_old = L2
            
            return 2,i
   


