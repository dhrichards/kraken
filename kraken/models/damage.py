import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type, nls
from kraken.numerics import maths_functions as mf
from kraken.numerics import energy_splits as es
from kraken.numerics.maths_functions import ε
from kraken.numerics import advection_numerics, solvers
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np


def initialise_damage(model):
    model.D = fem.functionspace(model.msh, ("Lagrange", 1))

    # self.H_el = bufl.quadrature_element(
    #     self.msh.basix_cell(), value_shape=(), scheme="default", degree=2
    # )
    model.H_space = fem.functionspace(model.msh, ("DG", 1))

    model.bc_d = []

    
    model.d = fem.Function(model.D, name="damage")
    model.Hprev = fem.Function(model.H_space, name="history")
    model.H = mf.history_function(model.ε_e, model.Hprev,
                                model.params.ν, model.params.ψcritstar)


def setup_damage_non_linear(model):



    C3 = model.params.C3; l = model.params.lstar

    H = mf.history_function(model.ε_e, model.Hprev,
                            model.params.ν, model.params.ψcritstar)
    

    v = ufl.TestFunction(model.D)


    F = (ufl.inner(model.d,v) + l**2*ufl.inner(ufl.grad(model.d), ufl.grad(v)) \
            - C3*l*2*(1-model.d)*H*v) * ufl.dx
    J = ufl.derivative(F,model.d,ufl.TrialFunction(model.D))


    model.damage_problem = solvers.SNESProblem(F, model.d, bcs=model.bc_d)

    model.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)

    model.damage_solver.setFunction(model.damage_problem.F, fem.petsc.create_vector(fem.form(F)))
    model.damage_solver.setJacobian(model.damage_problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)


    
    model.damage_solver.setType("newtonls")

    
    
    model.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
    model.damage_solver.getKSP().setType("preonly")
    model.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
    model.damage_solver.getKSP().getPC().setType("lu")
    # model.damage_solver.getKSP().getPC().setFactorSolverType("mumps")

    
   
def setup_damage_bounded(model, w=lambda d: d):

        C3 = model.params.C3; l = model.params.lstar
        ψcrit = model.params.ψcritstar; ν = model.params.ν

        model.d_lb = fem.Function(model.D, name="damage_lb")
        model.d_ub = fem.Function(model.D, name="damage_ub")
        model.d_lb.x.array[:] = 0.0
        model.d_ub.x.array[:] = 1.0
        

    
        s = np.linspace(0,1,500)
        c0 = 4*np.trapezoid(np.sqrt(w(s)),s)

        g = mf.degradation_default(model.d)

        H = ufl.max_value(es.free_energy_plus_spectral(model.ε_e,ν) - ψcrit,0)


      
        

        dissipated_energy = (1/C3) * mf.crack_density_function(model.d,l,w, c0)*ufl.dx
        elastic_energy = g * H * ufl.dx
       
        total_energy = dissipated_energy + elastic_energy #- self.external_energy_without_surface()



        F = ufl.derivative(total_energy,model.d,ufl.TestFunction(model.D))
        J = ufl.derivative(F,model.d,ufl.TrialFunction(model.D))

        model.damage_problem = solvers.SNESProblem(F, model.d, bcs=model.bc_d)

        model.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)

        model.damage_solver.setFunction(model.damage_problem.F, fem.petsc.create_vector(fem.form(F)))
        model.damage_solver.setJacobian(model.damage_problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)


        
        model.damage_solver.setType("vinewtonrsls")
        model.damage_solver.setVariableBounds(model.d_lb.x.petsc_vec,model.d_ub.x.petsc_vec)
        

        
        model.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
        model.damage_solver.getKSP().setType("preonly")
        model.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
        model.damage_solver.getKSP().getPC().setType("lu")




def setup_damage_linear(model):

    C3 = model.params.C3; l = model.params.lstar
    
    d = ufl.TrialFunction(model.D)
    v = ufl.TestFunction(model.D)

    H = mf.history_function(model.ε_e, model.Hprev,
                            model.params.ν, model.params.ψcritstar)


    F = (ufl.inner(d,v) + l**2*ufl.inner(ufl.grad(d), ufl.grad(v)) \
            - C3*l*2*(1-d)*H*v) * ufl.dx
    
    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    A = fem.petsc.assemble_matrix(a, bcs=model.bc_d)
    A.assemble()
    b = fem.petsc.create_vector(L)

    model.damage_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    model.damage_solver.setOperators(A)
    model.damage_solver.setType("preonly")
    model.damage_solver.getPC().setType("lu")

    # model.damage_problem = fem.petsc.LinearProblem(a, L, bcs=model.bc_d,
    #     petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    

 
