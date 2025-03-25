#%%
import numpy as np
from matplotlib import pyplot as plt


d = np.linspace(0,1,100)
def deg(d,q):
    ϕ = 1-d
    return (q+1)*(1 - (q/(q+1))**(ϕ**2) )


for q in np.logspace(-17,4,9):
    plt.plot(d,deg(d,q),label=str(q))


plt.legend()

#%%

def DgDd(d,q):
    ϕ = 1-d
    a = q/(q+1)
    return 2*(q+1)*(1-d)*a**(ϕ**2)*np.log(a)

def DgDdtaylor(d,q):
    ϕ = 1-d
    a = q/(q+1)
    return 2*(q+1)*(1-d)*np.log(a)*(1+ϕ**2*np.log(a))
q = 200

# g = deg(d,q)
g = (1-d)**2 + 1e-5
dg = np.gradient(g,d)


plt.plot(d,g)
plt.plot(d,dg)
plt.plot(d,-2*(1-d))
# plt.plot(d,DgDd(d,q))
# plt.plot(d,DgDdtaylor(d,q))

#%%

q = np.logspace(-10,4,100)
lp = q*np.log((q+1)/q)

plt.plot(q,lp)
# log log
plt.xscale("log")
plt.yscale("log")

plt.xlabel("q")
plt.ylabel("l0/l")