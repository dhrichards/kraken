import ufl
from dolfinx import fem, default_scalar_type
from kraken.numerics.energy_splits import λoverμ

def deviatoric_stress(ε, σD_prev, η, De):
    return 1/(1 + De*η) * (2*η*ε + De *η* σD_prev) 


def elastic_strain(σD, p, ν):
    D = ufl.shape(σD)[0]
    I = ufl.Identity(D)
    return 0.5*σD - p/(D*(λoverμ(ν)+2/D)) * I

# def viscosity(σD, n=3.0, eps=1.e-11): 
#     τe2 = ufl.inner(σD, σD) / 2
#     return 0.5 * (τe2 + eps)**(1-n)



def ucm_steady(σD, u):
    # steady part of the upper convected Maxwell model
    i, j, k = ufl.indices(3)
    return u[k]*ufl.Dx(σD[i,j],k) \
            - ufl.Dx(u[i],k)*σD[k,j] \
            - ufl.Dx(u[j],k)*σD[i,k] 


