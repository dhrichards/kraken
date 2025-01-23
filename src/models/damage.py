import basix.ufl as bufl
import ufl
from dolfinx import fem, default_real_type
from ..numerics import maths_functions as mf
from ..numerics.maths_functions import ε
from ..numerics import advection_numerics


class DamageSolver:
    def __init__(self, msh, bc_func, material):
        self.msh = msh
        self.material = material

        d_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        self.V = fem.functionspace(self.msh, d_el)

        self.bcs = bc_func(self.V)

    def solve(self,v,Hprev):

        H = mf.history_function(ε(v),Hprev,self.material.ν,self.material.ψcritstar)

        C3 = self.material.C3; l = self.material.l

        d, g = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        F = (ufl.inner(d,g) + l**2*ufl.inner(ufl.grad(d), ufl.grad(g)) \
                - C3*l*2*(1-d)*H*g) * ufl.dx
        
        a, L = ufl.lhs(F), ufl.rhs(F)

        self.problem = fem.petsc.LinearProblem(a, L, self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        return self.problem.solve()
    
class HistorySolver:
    def __init__(self, msh, material, dt=0.0):
        self.msh = msh
        self.material = material
        self.dt = dt
        

        h_el = bufl.element("DG", self.msh.basix_cell(), 0, dtype=default_real_type)
        self.V = fem.functionspace(self.msh, h_el)
        
    def solve(self, Hprev, v):


        h, g = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        H = mf.history_function(ε(v),Hprev,self.material.ν,self.material.ψcritstar)

        a = ufl.inner(h,g) * ufl.dx
        L = ufl.inner(H,g) * ufl.dx

        self.problem = fem.petsc.LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        return self.problem.solve()
    

    def advect(self, Hprev, u, k=1e-4):

        f = ufl.TrialFunction(self.V)
        g = ufl.TestFunction(self.V)
        
        n = ufl.FacetNormal(self.msh)
        h = ufl.CellDiameter(self.msh)

        α = 3.0

        Dt = lambda f: advection_numerics.backward_euler(f, Hprev, self.dt)
        a_A = lambda f, g: advection_numerics.advection(f, g, u, n)
        a_D = lambda f, g: advection_numerics.diffusion(f, g, k, α, n, h)

        F = Dt(f)*g*ufl.dx + a_A(f,g)+ a_D(f,g)

        a, L = ufl.lhs(F), ufl.rhs(F)

        problem = fem.petsc.LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        return problem.solve()





        

    




def damage_solver(V, bc_func, material, v, Hprev):
        

    bcs = bc_func(V)


    H = mf.history_function(ε(v),Hprev,material.ν,material.ψcritstar)
    # H = solve_history()

    d = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    C3 = material.C3; l = material.l

    F = (ufl.inner(d,v) + l**2*ufl.inner(ufl.grad(d), ufl.grad(v)) \
            - C3*l*2*(1-d)*H*v) * ufl.dx
    
    a, L = ufl.lhs(F), ufl.rhs(F)

    damage_problem = fem.petsc.LinearProblem(a, L, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    
    return damage_problem

def history_solver(msh, material, Hprev, v):
    h_el = bufl.element("DG", msh.basix_cell(), 0, dtype=default_real_type)
    Q_h = fem.functionspace(msh, h_el)

    h, g = ufl.TrialFunction(Q_h), ufl.TestFunction(Q_h)

    H = mf.history_function(ε(v),Hprev,material.ν,material.ψcritstar)

    a = ufl.inner(h,g) * ufl.dx
    L = ufl.inner(H,g) * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    return problem

