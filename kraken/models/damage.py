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


def setup_damage_non_linear(model,free_energy_plus=es.free_energy_plus_dp):



    C3 = model.params.C3; l = model.params.lstar

    H = es.history_function(model.ε_e, model.Hprev,
                            model.params.ν, model.params.ψcritstar, free_energy_plus)
    

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

    
def setup_higher_order_spaces(model,bc_func):
     
     
    model.d_el = bufl.element("CG", model.msh.basix_cell(), 1)

    model.d_el_mixed = bufl.mixed_element([model.d_el, model.d_el])
    model.D_mixed = fem.functionspace(model.msh, model.d_el_mixed)
    model.d_mixed = fem.Function(model.D_mixed, name="damage_mixed")

    
    model.D, _ = model.D_mixed.sub(0).collapse()

    model.d, model.lap = ufl.split(model.d_mixed)

    model.H_space = fem.functionspace(model.msh, ("DG", 1))   
    model.Hprev = fem.Function(model.H_space, name="history")

    model.g = es.degradation_default(model.d)

    model.bc_d = bc_func(model.D_mixed)


def setup_damage_higher_order(model, free_energy_plus=es.free_energy_plus_spectral):
        
    C3 = model.params.C3; l = model.params.lstar

    l0 = l/2
    c = 1-model.d
    
    H = es.history_function(model.ε_e, model.Hprev,
                            model.params.ν, model.params.ψcritstar, free_energy_plus)
    
    # H = ufl.max_value(free_energy_plus(model.ε_e,model.params.ν) - model.params.ψcritstar,0) 

    mixed_test = ufl.TestFunction(model.D_mixed)
    v, q = ufl.split(mixed_test)   

    # F = (model.d*v - 0.5*l**2*model.lap*v - (1/16)*l**4*ufl.inner(ufl.grad(model.lap),ufl.grad(v)) \
    #         - C3*l*2*(1-model.d)*H*v) * ufl.dx \
    #         + (model.lap*q + ufl.inner(ufl.grad(model.d), ufl.grad(q))) * ufl.dx 

    F = (C3*4*l0*c*v*H + c*v - 2*l0**2*model.lap*v - l0**4*ufl.inner(ufl.grad(model.lap), ufl.grad(v)) \
            -1.0*v ) * ufl.dx \
            - (model.lap*q + ufl.inner(ufl.grad(c), ufl.grad(q))) * ufl.dx
            
    J = ufl.derivative(F,model.d_mixed,ufl.TrialFunction(model.D_mixed))

    model.damage_problem = solvers.SNESProblem(F, model.d_mixed, bcs=model.bc_d)

    model.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)

    model.damage_solver.setFunction(model.damage_problem.F, fem.petsc.create_vector(fem.form(F)))
    model.damage_solver.setJacobian(model.damage_problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)



    model.damage_solver.setType("newtonls")



    model.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
    model.damage_solver.getKSP().setType("preonly")
    model.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
    model.damage_solver.getKSP().getPC().setType("lu")
    # model.damage_solver.getKSP().getPC().setFactorSolverType("mumps")


                    







   
def setup_damage_bounded(model, w=lambda d: d, free_energy_plus=es.free_energy_plus_spectral):

    C3 = model.params.C3; l = model.params.lstar
    ψcrit = model.params.ψcritstar; ν = model.params.ν

    d_ub = fem.Function(model.D, name="damage_ub")
    d_ub.x.array[:] = 1.0



    s = np.linspace(0,1,500)
    c0 = 4*np.trapezoid(np.sqrt(w(s)),s)

    H = ufl.max_value(free_energy_plus(model.ε_e,ν) - ψcrit,0) 

    # R = es.cauchy_stress(model.ε_e, ν) + mf.water_pressure(model.msh,model.u)*ufl.Identity(model.msh.geometry.dim)
    # H = mf.clayton_driving_function(R, model.params.σcritstar)





    dissipated_energy = (1/C3) * es.crack_density_function(model.d,l,w, c0)*ufl.dx
    elastic_energy = model.g * H * ufl.dx
    # pressure_work = -pw*ufl.inner(ufl.grad(g), model.u) * ufl.dx

    total_energy = dissipated_energy + elastic_energy 



    F = ufl.derivative(total_energy,model.d,ufl.TestFunction(model.D))
    J = ufl.derivative(F,model.d,ufl.TrialFunction(model.D))

    model.damage_problem = solvers.SNESProblem(F, model.d, bcs=model.bc_d)

    model.damage_solver = PETSc.SNES().create(MPI.COMM_WORLD)

    model.damage_solver.setFunction(model.damage_problem.F, fem.petsc.create_vector(fem.form(F)))
    model.damage_solver.setJacobian(model.damage_problem.J, fem.petsc.create_matrix(fem.form(J)),P=None)



    model.damage_solver.setType("vinewtonrsls")
    model.damage_solver.setVariableBounds(model.d_prev_time.x.petsc_vec,d_ub.x.petsc_vec)



    model.damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
    model.damage_solver.getKSP().setType("cg")
    model.damage_solver.getKSP().setTolerances(rtol=1.0e-9)
    model.damage_solver.getKSP().getPC().setType("jacobi")
    model.damage_solver.getKSP().getPC().setFactorSolverType("mumps")




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
    

 
