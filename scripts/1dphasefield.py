#%%
import numpy as np
from matplotlib import pyplot as plt

ε = np.linspace(0,0.0005,300)
E = 9.33e9
Gc = 1.0
l = 1e-2

σ = (Gc/(Gc + E*ε**2*l))**2 * E * ε


fig, ax = plt.subplots()
ax.plot(ε, σ)


#%%
import sympy as sp

E = sp.symbols('E', positive=True)
Gc = sp.symbols('G_c', positive=True)
l = sp.symbols('l', positive=True)
ε = sp.symbols('epsilon', positive=True)
ψcrit = sp.symbols('psi_crit', positive=True)



def positive_part(x):
    return (x + sp.Abs(x))/2

## phase field equation
d = sp.symbols('d', positive=True)
g = (1 - d)**2
σ = g * E * ε
# ψ = 0.5*σ**2 / E
ψ = 0.5*E*ε**2

# H = positive_part(ψ - ψcrit)
H = ψ

eq = sp.Eq(Gc*d/l, 2*(1-d)*H)
sol = sp.solve(eq, d)
# print(sol)
d_sol = sp.simplify(sol[0])