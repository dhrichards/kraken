import adios4dolfinx
from .base import Momentum
import numpy as np
from dolfinx import fem, mesh, la
import dolfinx
from mpi4py import MPI
import ufl
import basix.ufl as bufl
import numpy as np
from kraken import parameters
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics import solvers
from petsc4py import PETSc
from kraken.numerics.invariants import matrix_function
from kraken.boundaryconditions import marked_ds
from time import perf_counter


class Picard(Momentum):
    '''
    Class for solving the momentum equation for a Maxwell viscoelastic material

    Time evolution is handled using a semi-Lagrangian approach, such that it solves for the change
    in displacements at each timestep and the mesh is then moved. 

    Solves for the change in total displacement, change in viscous displacement, and pressure as a mixed function.
    The elastic displacement is then calculated as the difference between the total and viscous displacements.
    The elastic strain is then calculated from the elastic displacement and the previous timestep's elastic strain.

    The viscosity can be non-newtonian, for some power law
    '''

    def __init__(self, sim):
        super().__init__(sim)

        self.elastic_time =0.0
        self.viscous_time = 0.0

        self.mesh_smoothing = False

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        self.mixed_el = bufl.mixed_element([self.u_el, self.p_el])

        self.U = fem.functionspace(self.sim.msh, self.u_el)
        self.W = fem.functionspace(self.sim.msh, self.mixed_el)


        self.du_e = fem.Function(self.U, name="change in total displacement")
        self.w = fem.Function(self.W, name="mixed function")

        self.du_v, self.dp = ufl.split(self.w)
        # self.du_e = self.du - self.du_v
        self.du = self.du_v + self.du_e
        # self.w.x.array[:] =1.0


        self.du_e_prev_it = fem.Function(self.U, name="total displacement prev it")
        self.w_prev_it = fem.Function(self.W, name="mixed function previous iteration")

        self.du_v_prev_it, self.dp_prev_it = ufl.split(self.w_prev_it)
        # self.du_e_prev_it = self.du_prev_it - self.du_v_prev_it
        self.du_prev_it = self.du_e_prev_it + self.du_v_prev_it

        
        self.w_prev_time = fem.Function(self.W, name="mixed function previous time")
        self.u_e_prev_time = fem.Function(self.U, name="total displacement prev time")

        self.u_v_prev_time, self.p_prev_time = ufl.split(self.w_prev_time)
        # self.u_e_prev_time = self.u_prev_time - self.u_v_prev_time
        self.u_prev_time = self.u_v_prev_time + self.u_e_prev_time
        
        self.bc_e = self.sim.bc_funcs[0][0](self.U)
        self.bc_s = self.sim.bc_funcs[0][1](self.W)

        

        
        self.ε_el = bufl.element("DG", self.sim.msh.basix_cell(), 1, shape=(self.sim.msh.geometry.dim, self.sim.msh.geometry.dim))
        self.E = fem.functionspace(self.sim.msh, self.ε_el)

        self.ε_e_prev_time = fem.Function(self.E, name="epsiloneprevtime")

    

        

        


    def setup_momentum(self):
        self.setup_elastic()
        self.setup_stokes()


    def setup_elastic(self):


        v = ufl.TestFunction(self.U)
        n = ufl.FacetNormal(self.sim.msh)



        self.ρ = self.sim.params.ρistar/self.area_ratio
        f = self.ρ*mf.body_force(self.sim.msh)


        self.F_e = (
            + ufl.inner(self.σ, mf.ε(v)) - ufl.inner(f, v) 
              ) * ufl.dx 


        if self.sim.basal_friction:
            # water pressure on other boundaries
            self.F_e += (
                self.pw * ufl.inner(n, v) * self.sim.marked_ds(2)\
                )

            # linear weertman on bottom boundary
            self.F_e += (
                self.sim.params.βstar*ufl.inner(self.du/self.sim.params.dtstar,v)*self.sim.marked_ds(1)
            )
        else:
            self.F_e += (
                self.pw * ufl.inner(n, v) * ufl.ds\
                )
        
        self.J_e = ufl.derivative(self.F_e,self.du_e,ufl.TrialFunction(self.U))
        
        self.problem_e = solvers.SNESProblem(self.F_e, self.du_e, bcs=self.bc_e)

    def setup_stokes(self):

            w_test = ufl.TestFunction(self.W)
            v_v, q = ufl.split(w_test)

    
            g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)
            
            σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
            
            if self.sim.params.n.value==1.0:
                η0 = 1.0
            else:
                η0 = mf.viscosity(ufl.dev(mf.ε(self.vel_prev_it)), self.sim.params.n, 1e-19)
            η = (1-self.sim.damage.d)**2*η0 + self.sim.params.viscosity_tol
        
    
            self.F_s = (
                    # η0*ufl.inner(εD, mf.ε(v_v))\
                    η*ufl.inner(mf.ε(self.vel), mf.ε(v_v))\
                    - g*ufl.inner(self.p, ufl.div(v_v))  \
                -    g*ufl.inner(σ0, mf.ε(v_v))
                    ) * ufl.dx
            
    
            self.F_s += (
                    - g*ufl.div(self.du_v)*q \
                    ) * ufl.dx 
        
    
            self.J_s = ufl.derivative(self.F_s,self.w,ufl.TrialFunction(self.W))
            
            self.problem_s = solvers.SNESProblem(self.F_s, self.w, bcs=self.bc_s)
            
    

        
    def setup_solver(self):
    
    
            self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
            self.solver_s = PETSc.SNES().create(MPI.COMM_WORLD)

            
            for solver in [self.solver,self.solver_s]:
                solver.setTolerances(rtol=1.0e-11, max_it=10, atol=1e-13)
                solver.getKSP().setType("preonly")
                # solver.getKSP().setTolerances(rtol=1.0e-7)
                solver.getKSP().getPC().setType("lu")
                solver.getKSP().getPC().setFactorSolverType("mumps")
     
            # self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.extract_function_spaces(fem.form(self.F))))
            self.solver.setFunction(self.problem_e.F,dolfinx.fem.petsc.create_vector(fem.form(self.F_e)))
            self.solver.setJacobian(self.problem_e.J,dolfinx.fem.petsc.create_matrix(fem.form(self.J_e)),P=None)

            self.solver_s.setFunction(self.problem_s.F,dolfinx.fem.petsc.create_vector(fem.form(self.F_s)))
            self.solver_s.setJacobian(self.problem_s.J,dolfinx.fem.petsc.create_matrix(fem.form(self.J_s)),P=None)            


    def solve(self):

        comm = MPI.COMM_WORLD

        comm.Barrier()
        t0 = perf_counter()

        self.solver.solve(None,self.du_e.x.petsc_vec)
        self.du_e.x.scatter_forward()

        comm.Barrier()
        t1 = perf_counter()

        self.solver_s.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()

        comm.Barrier()
        t2 = perf_counter()

        self.w_prev_it.x.array[:] = self.w.x.array[:]
        self.du_e_prev_it.x.array[:] = self.du_e.x.array[:]

        self.elastic_time += t1 - t0
        self.viscous_time += t2 - t1

    def timestep(self):

        self.ε_e_prev_time.interpolate(fem.Expression(self.ε_e, self.E.element.interpolation_points()))


        du = fem.Function(self.V)
        du.interpolate(fem.Expression(self.du,self.V.element.interpolation_points()))

        self.sim.msh.geometry.x[:,:self.sim.msh.geometry.dim] += self.sim.params.ucstar_float*du.x.array.reshape((-1, self.sim.msh.geometry.dim))
        
        self.w_prev_time.x.array[:] += self.w.x.array[:]
        self.u_e_prev_time.x.array[:] += self.du_e.x.array[:]

        self.area = fem.assemble_vector(self.cell_area_form).array
        self.area_ratio.x.array[:] = self.area/self.area_0

        self.elastic_time = 0
        self.viscous_time = 0
        

        
    def write_checkpoint(self, filename, t=0):
        super().write_checkpoint(filename, t)
        adios4dolfinx.write_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

    def read_checkpoint(self, filename, t=0):
        super().read_checkpoint(filename, t)
        adios4dolfinx.read_function(filename, self.ε_e_prev_time, name="epsiloneprevtime", time=t)

        
        
class PicardNested(Picard):
    def __init__(self, sim):
        super().__init__(sim)

        self.u_el = bufl.element("CG", self.sim.msh.basix_cell(), 2, shape=(self.sim.msh.geometry.dim,))
        self.p_el = bufl.element("CG", self.sim.msh.basix_cell(), 1)

        
        self.U = fem.functionspace(self.sim.msh, self.u_el)
        self.P = fem.functionspace(self.sim.msh, self.p_el)


        self.du_e = fem.Function(self.U, name="change in elastic displacement")
        self.du_v = fem.Function(self.U, name="change in viscous displacement")
        self._p = fem.Function(self.P)

        self.du = self.du_v + self.du_e

        self.du_e_prev_it = fem.Function(self.U)
        self.du_v_prev_it = fem.Function(self.U)
        self._p_prev_it = fem.Function(self.P)

        self.du_prev_it = self.du_e_prev_it + self.du_v_prev_it

    
        self.u_e_prev_time = fem.Function(self.U)
        self.u_v_prev_time = fem.Function(self.U)
        self.p_prev_time = fem.Function(self.P)

        self.u_prev_time = self.u_v_prev_time + self.u_e_prev_time
        
        self.bc_e = self.sim.bc_funcs[0][0](self.U)
        self.bc_s = self.sim.bc_funcs[0][1](self.U)

    @property
    def p(self):
        return self._p

    @property
    def p_prev_it(self):
        return self._p_prev_it

    

    
    def setup_stokes(self):

        (δu, p) = ufl.TrialFunction(self.U), ufl.TrialFunction(self.P)
        (v, q) = ufl.TestFunction(self.U), ufl.TestFunction(self.P)


        g = es.degradation(self.sim.damage.d,self.sim.params.ge_tol)

        σ0 = es.cauchy_stress(self.ε_e, self.sim.params.ν)
        
        if self.sim.params.n.value==1.0:
            η0 = 1.0
        else:
            η0 = mf.viscosity(ufl.dev(mf.ε(self.vel_prev_it)), self.sim.params.n, 1e-19)
        η = (1-self.sim.damage.d)**2*η0 + self.sim.params.viscosity_tol
    
        # p = δp + self.p_prev_time

        self.a = fem.form([[η*ufl.inner(mf.ε(δu/self.sim.params.dtstar), mf.ε(v)) * ufl.dx, g*ufl.inner(p, ufl.div(v)) * ufl.dx], 
                           [g*ufl.inner(ufl.div(δu/self.sim.params.dtstar), q) * ufl.dx, None]])
        # self.a = fem.form([[η*ufl.inner(mf.ε(δu/self.sim.params.dtstar), mf.ε(v)) * ufl.dx, - g*ufl.inner(p, ufl.div(v))], 
        #               [-g*ufl.div(δu/self.sim.params.dtstar)*q * ufl.dx, None]])
        self.L = fem.form([g*ufl.inner(σ0, mf.ε(v)) * ufl.dx, ufl.inner(fem.Constant(self.sim.msh, PETSc.ScalarType(0)), q) * ufl.dx]) 


        self.a_p11 = fem.form(η**-1*ufl.inner(p, q) * ufl.dx)
        self.a_p = [[self.a[0][0], None], [None, self.a_p11]]

        # self.F_s = [
        #         (
        #         # η0*ufl.inner(εD, mf.ε(v_v))\
        #         η*ufl.inner(mf.ε(δu/self.sim.params.dtstar), mf.ε(v))\
        #         - g*ufl.inner(p, ufl.div(v))  \
        #     -    g*ufl.inner(σ0, mf.ε(v))
        #         ) * ufl.dx,
        #             (
        #         - g*ufl.div(self.δu/self.sim.params.dtstar)*q \
        #         ) * ufl.dx  
        #         ]

        

        # self.J_s = [[ufl.derivative(self.F_s[0],self.du_v,δu), ufl.derivative(self.F_s[0],self.dp,δp)],
        #             [ufl.derivative(self.F_s[1],self.du_v,δu), ufl.derivative(self.F_s[1],self.dp,δp)]]

        # self.P_s = [[self.J_s[0][0], None],
        #             [None, η**-1 * δp * q * ufl.dx]]
        


    
    def setup_solver(self):
        self.setup_elastic_solver()
        self.setup_viscous_solver()

    def setup_elastic_solver(self):

        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
            
        self.solver.setTolerances(rtol=1.0e-11, max_it=10, atol=1e-13)
        self.solver.getKSP().setType("preonly")
        # # self.solver.getKSP().setTolerances(rtol=1.0e-7)
        self.solver.getKSP().getPC().setType("lu")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")

       
     
        # self.solver.setFunction(self.problem.F,dolfinx.fem.petsc.create_vector(fem.extract_function_spaces(fem.form(self.F))))
        self.solver.setFunction(self.problem_e.F,dolfinx.fem.petsc.create_vector(fem.form(self.F_e)))
        self.solver.setJacobian(self.problem_e.J,dolfinx.fem.petsc.create_matrix(fem.form(self.J_e)),P=None)
        # nullspace = build_nullspace(self.U)
        # A = dolfinx.fem.petsc.create_matrix(fem.form(self.J_e))
        # A.setNearNullSpace(nullspace)
        # self.solver.setJacobian(
        #     self.problem_e.J,
        #     A,
        #     P=None
        # )


    def setup_viscous_solver(self):

        # Assemble nested matrix operators
        A = fem.petsc.assemble_matrix_nest(self.a, bcs=self.bc_s)
        A.assemble()

        # Create a nested matrix P to use as the preconditioner. The
        # top-left block of P is shared with the top-left block of A. The
        # bottom-right diagonal entry is assembled from the form a_p11:
        P11 = fem.petsc.assemble_matrix(self.a_p11, [])
        P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
        P.assemble()

        A00 = A.getNestSubMatrix(0, 0)
        A00.setOption(PETSc.Mat.Option.SPD, True)

        P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)
        P00.setOption(PETSc.Mat.Option.SPD, True)
        P11.setOption(PETSc.Mat.Option.SPD, True)

        # Assemble right-hand side vector
        self.b = fem.petsc.assemble_vector_nest(self.L)

        # Modify ('lift') the RHS for Dirichlet boundary conditions
        fem.petsc.apply_lifting_nest(self.b, self.a, bcs=self.bc_s)

        # Sum contributions for vector entries that are share across
        # parallel processes
        for b_sub in self.b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set Dirichlet boundary condition values in the RHS vector
        bcs0 = fem.bcs_by_block(fem.extract_function_spaces(self.L), self.bc_s)
        fem.petsc.set_bc_nest(self.b, bcs0)

        # The pressure field is determined only up to a constant. We supply
        # a vector that spans the nullspace to the solver, and any component
        # of the solution in this direction will be eliminated during the
        # solution process.
        # null_vec = fem.petsc.create_vector_nest(self.L)

        # # Set velocity part to zero and the pressure part to a non-zero
        # # constant
        # null_vecs = null_vec.getNestSubVecs()
        # null_vecs[0].set(0.0), null_vecs[1].set(1.0)

        # # Normalize the vector that spans the nullspace, create a nullspace
        # # object, and attach it to the matrix
        # null_vec.normalize()
        # nsp = PETSc.NullSpace().create(vectors=[null_vec])
        # assert nsp.test(A)
        # A.setNullSpace(nsp)

        # Create a MINRES Krylov solver and a block-diagonal preconditioner
        # using PETSc's additive fieldsplit preconditioner
        self.solver_s = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver_s.setOperators(A, P)
        self.solver_s.setType("minres")
        self.solver_s.setTolerances(rtol=1e-7)
        self.solver_s.getPC().setType("fieldsplit")
        self.solver_s.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

        # Define the matrix blocks in the preconditioner with the velocity
        # and pressure matrix index sets
        nested_IS = P.getNestISs()
        self.solver_s.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

        # Set the preconditioners for each block. For the top-left
        # Laplace-type operator we use algebraic multigrid. For the
        # lower-right block we use a Jacobi preconditioner. By default, GAMG
        # will infer the correct near-nullspace from the matrix block size.
        ksp_u, ksp_p = self.solver_s.getPC().getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType("gamg")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("jacobi")

        # Create finite element {py:class}`Function <dolfinx.fem.Function>`s
        # for the velocity (on the space `V`) and for the pressure (on the
        # space `Q`). The vectors for `u` and `p` are combined to form a
        # nested vector and the system is solved.
        self.x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(self.du_v.x), la.create_petsc_vector_wrap(self.p.x)])


    def solve(self):

        comm = MPI.COMM_WORLD

        comm.Barrier()
        t0 = perf_counter()

        self.solver.solve(None,self.du_e.x.petsc_vec)
        self.du_e.x.scatter_forward()

        comm.Barrier()
        t1 = perf_counter()

        # Reassemble viscous RHS
        self.b = fem.petsc.assemble_vector_nest(self.L)

        fem.petsc.apply_lifting_nest(
            self.b, self.a, bcs=self.bc_s
        )

        for b_sub in self.b.getNestSubVecs():
            b_sub.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE
            )

        bcs0 = fem.bcs_by_block(
            fem.extract_function_spaces(self.L),
            self.bc_s
        )
        fem.petsc.set_bc_nest(self.b, bcs0)

        self.solver_s.solve(self.b, self.x)
        self.du_v.x.scatter_forward()
        self.p.x.scatter_forward()

        comm.Barrier()
        t2 = perf_counter()

        self.du_v_prev_it.x.array[:] = self.du_v.x.array[:]
        self.p_prev_it.x.array[:] = self.p.x.array[:]
        self.du_e_prev_it.x.array[:] = self.du_e.x.array[:]

        self.elastic_time += t1 - t0
        self.viscous_time += t2 - t1

    
            

            



    
def build_nullspace(V: fem.FunctionSpace):
    """Build PETSc nullspace for 2D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [
        la.vector(V.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType)
        for i in range(3)
    ]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x and y dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(2)]

    # Set the two translational rigid body modes
    b[0][dofs[0]] = 1.0
    b[1][dofs[1]] = 1.0

    # Set the rotational rigid body mode
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()

    x0 = x[dofs_block, 0]
    x1 = x[dofs_block, 1]

    b[2][dofs[0]] = -x1
    b[2][dofs[1]] = x0

    # Orthonormalize the basis
    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(
            x[: bs * length0],
            bsize=2,
            comm=V.mesh.comm
        )
        for x in b
    ]

    return PETSc.NullSpace().create(vectors=basis_petsc)