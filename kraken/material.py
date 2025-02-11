from kraken.utilities import mesh_sizes
import numpy as np
secperyr = 60*60*24*365.25

class Material_no_uc:
    def __init__(self, ρi=900, ρw=1000, g=9.81, E=9.33e9, ν=0.325, A=1.2e-25, n=3.0, Gc=1.0, ψcrit=1.0, σc = 0.1e6):

        # Default material properties
        self.ρi = ρi # Density of ice
        self.ρw = ρw # Density of water
        self.g = g # Gravitational acceleration
        self.E = E # Young's modulus
        self.ν = ν # Poisson's ratio
        self.A = A # Flow law parameter
        self.n = n # Flow law exponent
        self.Gc = Gc # Fracture toughness
        self.ψcrit = ψcrit 
        self.slope_angle = 0.0
        self.σc = σc # tensile strength

        # Choose L, τ such that C1, C2, C3*l = 1 by default
        self.L = self.μ/(self.ρw*self.g)
        self.τ = (self.A**(1/self.n) * self.μ)**-self.n
        self.l = 1/self.C3 # Regularisation length (non dimensional)

        self.Hc = self.μ
        self.ψcritstar = self.ψcrit / self.Hc
        self.pc = self.μ 
        self.ρratio = self.ρi/self.ρw

    @property
    def q(self):
        """Calculate the paramerter q for Lo et al. degradation model."""
        a = 3*self.Gc*self.E / (8*self.l*self.L*self.σc**2)
        q = 1.0
        for i in range(100):
            q = a/np.log((1+q)/q)
        return q
        

    @property
    def pwc(self):
        return self.ρw * self.g * self.L

    @property
    def λ(self):
        """Calculate the first Lamé parameter."""
        return self.ν * self.E / ((1 + self.ν) * (1 - 2 * self.ν))

    @property
    def μ(self):
        """Calculate the shear modulus (second Lamé parameter)."""
        return self.E / (2 * (1 + self.ν))
    
    @property
    def C1(self):
        """Non dimensional constant describing ratio between
        external and elastic stresses."""
        return self.L * self.ρw * self.g / (self.μ)
    
    @property
    def C2(self):
        """Non dimensional constant describing ratio between
        elastic and viscousc stresses."""
        return (self.A*self.τ)**(1/self.n) * self.μ 
    
    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture stresses."""
        return self.μ * self.L / self.Gc 
    


    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.l = factor*h.max()
        

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.τ
    
    def nondimt2yrs(self,t):
        return t*self.τ/secperyr



    



