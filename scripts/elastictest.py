#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
from dolfinx import io
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr


def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length/2)

def bottom_boundary(x):
    return np.isclose(x[1], -Hw)

def top_boundary(x):
    return np.isclose(x[1], nondim_height - Hw)

def all_boundaries(x):
    return left_boundary(x) + right_boundary(x) + bottom_boundary(x) + top_boundary(x)
# def crack(x):
#     x_c = nondim_length/2 - nondim_height
#     l = params.lstar
#     return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.9*nondim_height))# + (x[1]<(0.1-0.9*refineH[1]))


true_length = 16e3
true_height = 300


path = './outputs'
os.makedirs(path, exist_ok=True)


L = true_height
l = 1.0


nondim_length = true_length/L
nondim_height = true_height/L

Hw = 0.9


refineH = (1.4,0.3)
msh = kr.utilities.create_refined_mesh(nondim_length,nondim_height, l/L,
                                     aspect_ratios=(300,100), refine=refineH,
                                     cell_factor=1.0)
# msh.geometry.x[:,1] += 0.5

no_bc = lambda V: []
bc_d = lambda V: [bc.internal_bc(V.sub(0), fixed, 0.0),
                #   bc.get_zero_bc(V.sub(1), all_boundaries)
                  ]


u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]


model = kr.models.elasticity.elastic_damage(msh, [u_bc,bc_d])


model.params.L.value = L
model.params.l.value = l
model.params.dt.value = 60*60*2
model.params.ψcrit.value = 0.0
model.params.Gc.value = 1.0
model.params.patm.value = 0.0

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%
min_its = 5

model.setup_all()
# H = mf.clayton_driving_function(es.cauchy_stress(model.ε_e, model.params.ν), model.params.σcritstar, mf.water_pressure_static(model.msh))

gs = [2,4,6,8,9,9.8]


for i in range(len(gs)):
    model.params.g.value = gs[i]


    kr.iterators.fixed_point(model,min_its=min_its,max_its=50,tol=1e-6)
    # model.solve()

    kr.utilities.write_xdmf(path + "/elastictest" + str(i) + ".xdmf",
                            msh, [model.u,model.d,es.free_energy_plus_dplike(model.ε_e,model.params.ν)],
                            ["u","d","pp"], t=i)
    
    # model.d_prev_time.x.array[:] = model.d.x.array[:]
    

