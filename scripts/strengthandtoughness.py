#%%
import numpy as np
from matplotlib import pyplot as plt


E = 9.33e9
σc = 0.2e6
l = np.logspace(-1,1,100)
ν = 0.325

## Lower order AT1 model

Gc = 8*σc**2*l/(3*E)
ψc = 3*Gc/(16*l)
Kc = np.sqrt(Gc*E*(1-ν**2))

## Higher order AT1 model

Gc_ho = 2*l*σc**2/E
ψc_ho = Gc/(4*l)
Kc_ho = np.sqrt(Gc_ho*E*(1-ν**2))

fig, ax = plt.subplots()
ax.plot(l, Gc, label='Lower order AT1')
ax.plot(l, Gc_ho, label='Higher order AT1')
# ax.plot(l, ψc, label='ψc Lower order AT1')
ax.legend()
ax.set_xlabel('Length scale l (m)')
ax.set_ylabel('Gc ')


fig2, ax2 = plt.subplots()
ax2.plot(l, ψc, label='Lower order AT1')
ax2.plot(l, ψc_ho, label='Higher order AT1')
ax2.legend()
ax2.set_xlabel('Length scale l (m)')
ax2.set_ylabel('ψc ')    

fig3, ax3 = plt.subplots()
ax3.plot(l, Kc, label='Lower order AT1')
ax3.plot(l, Kc_ho, label='Higher order AT1')
ax3.legend()
ax3.set_xlabel('Length scale l (m)')
ax3.set_ylabel('Kc ')


fig4, ax4 = plt.subplots()
ax4.plot(Gc, Kc, label='Lower order AT1')   
ax4.plot(Gc_ho, Kc_ho, label='Higher order AT1')
ax4.legend()
ax4.set_xlabel('Gc ')
ax4.set_ylabel('Kc ')


#%%

ρi = 900 # Density of ice
ρw = 1000 # Density of water
g = 9.81 # Gravitational acceleration
E = 9.33e9 # Young's modulus
ν = 0.325 # Poisson's ratio
A = 1.2e-25 # Flow law parameter
n = 3.0 # Flow law exponent
Gc = 1.0 # Fracture toughness
L = 300 # Characteristic length
l = 0.5 # Regularisation length
# σcrit =  0.1e6 # tensile strength
ψcrit = 1.0 
dt = 31556926 # Characteristic time in seconds
patm = 0 # Atmospheric pressure


ρc = ρw
μ = E / (2 * (1 + ν))
uc = ρc * g * L**2 / μ
ucstar = uc/L
τ = A**(-1) * ucstar**(1-n) * μ**(-n)
γdot = ucstar / τ
ηc = A**(-1/n) * γdot**((1-n)/n)
dtstar = dt / τ
De = 1 / dtstar
Hc = μ*(ucstar)**2
ψcritstar = ψcrit / Hc
pc = μ * ucstar
pwc = ρc * g * L
ρistar = ρi/ρc
δstar = 1.0 - ρistar
patmstar = patm / pwc
lstar = l/L
λ = ν * E / ((1 + ν) * (1 - 2 * ν))
C2 = A**(1/n) * (uc/L)**(1-1/n) * μ * τ**(1/n)
C3 = μ * uc**2 / (Gc * L)


    
