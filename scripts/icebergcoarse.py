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

# def crack(x):
#     x_c = nondim_length/2 - nondim_height
#     l = params.lstar
#     return (x[0]>(x_c-l/3))*(x[0]<(x_c+l/3))*(x[1]>0)

def fixed(x):
    return (x[0]<(nondim_length/2 - refineH[0]*0.98*nondim_height))# + (x[0]>(nondim_length/2 - nondim_height/2))



true_length = 16e3
true_height = 300

L = true_height
l = 75
ρi = 900
ρf = 350
ρsw = 1000
D = 32.5


path = './outputs'
os.makedirs(path, exist_ok=True)


nondim_length = true_length/L
nondim_height = true_height/L

# flotation_height = mf.flotation_height(ρi/ρsw,ρf/ρsw,D/L)
flotation_height = ρi/ρsw

refineH = (3.5,0.4)
msh = kr.utilities.create_refined_mesh(nondim_length, nondim_height, l/L, flotation_height,
                                     aspect_ratios=(1,1), refine=refineH,
                                     cell_factor=1.5)
# msh.geometry.x[:,1] += 0.5

d_bc = lambda V: []#bc.internal_bc(V, fixed, 0.0)]

u_bc = lambda V: [bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                           bc.get_zero_bc(V.sub(1).sub(0), left_boundary)
                        ]

# u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

# model = kr.models.jakub2.viscoelastic_damage(msh, [u_bc,d_bc])
model = kr.base.Simulation(msh, [u_bc, d_bc],
                           kr.momentum.mixed.SemiLagrangian,
                           kr.damage.higherorder.HigherOrder)
model.damage.free_energy_plus = es.free_energy_plus_dp
model.params.L = L
model.params.l = l
# model.params.n = 1
# model.params.A = model.params.A**(1/3)
model.params.dt = 60*60*24*30
model.params.ρi = ρi
model.params.ρw = ρsw
model.params.ψcrit = 1.0
model.params.Gc = 1.0
model.params.patm = 0.0

# model = oc.viscoelastic_damage(msh, [symm_bc,symm_bc,bc_d], kp.Params_no_uc(), 
#                                dt = 1.0)#g = lambda d: mf.degradation_Lo2023(d,0.05))


#%%
min_its = 10

# model.setup_all()
model.setup()
gs = [9.8]

for i,g in enumerate(gs):

    model.params.g = g

    # kr.iterators.fixed_point(model, min_its=min_its, tol=1e-5)
    model.fixed_point(min_its=min_its, tol=1e-5,max_its=50, solve_damage=False)

    # kr.utilities.write_xdmf(path + "/iceberggravity" + str(i) + ".xdmf",
    #                         msh, [model.u, model.d,
    #                               model.u_e, model.u_v,
    #                               ufl.tr(mf.ε(model.u)), ufl.tr(mf.ε(model.u_v)), ufl.tr(model.ε_e)],
    #                               ["u", "d",
    #                                "ue", "uv",
    #                                "tr_u", "tr_uv", "ε_e"],
    #                               t=i)
    η = mf.viscosity(mf.ε(model.momentum.du_v)/model.params.dtstar, model.params.n, 1.e-8)
    # η = mf.viscosity(model.momentum.ε_v/model.params.dtstar, model.params.n,0)
    # τv = η*mf.εD(model.momentum.u_v)
    # σv = η*(model.momentum.ε_v/model.params.dtstar) - model.momentum.p*ufl.Identity(2)
    τv = η*mf.ε(model.momentum.du_v)/model.params.dtstar
    
    σv = τv - model.momentum.p*ufl.Identity(2)
    
    
    σe = (model.momentum.stress(model.momentum.ε_e))
    τe = ufl.dev(σe)

    η2 = mf.viscosity_stress(τv, 3,1e-20)

    kr.utilities.write_xdmf(path + "/iceberggravity" + str(i) + ".xdmf",
                            msh, [model.momentum.u,model.damage.d,
                                #   model.momentum.u_e, model.momentum.u_v,
                                    η, σv, σe,η2,
                                    τv, τe,
                                    -model.momentum.p,ufl.tr(σe),η/(η2*ufl.inner(τv,τv)),
                                    ufl.div(σv), ufl.div(σe),
                                    # ufl.tr(model.momentum.ε_v)
                                    ufl.div(model.momentum.du_v),
                                    es.free_energy_plus_spectral(model.momentum.ε_e,model.params.ν),mf.viscous_energy(model.momentum.du_v/model.params.dtstar,model.params.n)
                                ],
                                  ["u","d",
                                # "ue","uv",
                                "η","σv","σe","η2",
                                "τv","τe",
                                "minusp","tr_σe","ratio",
                                "div_σv","div_σe",
                                "div_ε_v",
                                "ψe","ψv"
                                ],
                                  t=i)
    model.damage.timestep()
    # model.d_prev_time.x.array[:] = model.d.x.array[:]
    
