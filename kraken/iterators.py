import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem




def fixed_point(model, max_its=100, tol=1e-4, min_its=2, solve_damage=True):
        L2_old = 0.0

        one = fem.Function(model.D)
        one.x.array[:] = 1.0
        area = fem.assemble_scalar(fem.form(ufl.inner(one,one)*ufl.dx))

        area = np.sqrt(MPI.COMM_WORLD.allreduce(area, op=MPI.SUM))

        error_prev = 100
        
        for i in range(max_its):
            
            if solve_damage:
                model.solve_damage()
            model.solve()
   
            

            L2_ = ufl.inner(model.d,model.d)*ufl.dx
            L2_rank = fem.assemble_scalar(fem.form(L2_))
            L2 = np.sqrt(MPI.COMM_WORLD.allreduce(L2_rank, op=MPI.SUM))

            error_L2 = np.abs(L2 - L2_old)/area
            if MPI.COMM_WORLD.rank == 0:
                print(f"iteration {i}, error {error_L2}")

            if i>min_its-1:
                if (error_L2 < tol) and (error_prev < tol):
                    break
            
            error_prev = error_L2
            L2_old = L2

        # Update history function as finished fixed point iteration
        if solve_damage:
            model.update_history()
    
        
