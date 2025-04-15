from kraken.utilities import mesh_sizes
import numpy as np
secperyr = 60*60*24*365.25


class Material_no_uc:
    def __init__(self):

        # Default material properties
        self.ρi = 900 # Density of ice
        self.ρw = 1000 # Density of water
        self.g = 9.81 # Gravitational acceleration
        self.E = 9.33e9 # Young's modulus
        self.ν = 0.325 # Poisson's ratio
        self.A = 1.2e-25 # Flow law parameter
        self.n = 3.0 # Flow law exponent
        self.Gc = 1.0 # Fracture toughness
        self.L = 100 # Characteristic length
        self.l = 0.5 # Regularisation length
        self.slope_angle = 0.0
        self.σc =  0.1e6 # tensile strength
        self.ψcrit = 1.0 
        self.τ = secperyr # Characteristic time in seconds
        self.patm = 1e5 # Atmospheric pressure



    @property
    def q(self):
        """Calculate the paramerter q for Lo et al. degradation model."""
        a = 3*self.Gc*self.E / (8*self.l*self.σc**2)
        q = 1.0
        for i in range(100):
            q = a/np.log((1+q)/q)
        return q
        
    @property
    def ψcritstar(self):
        return self.ψcrit / self.μ
    
    @property
    def pwc(self):
        return self.ρw * self.g * self.L
    
    @property
    def ρratio(self):
        return self.ρi/self.ρw

    @property
    def lstar(self):
        return self.l/self.L

    @property
    def patmstar(self):
        return self.patm / self.pwc

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
        return self.L * self.ρw * self.g /  self.μ

    @property
    def C2(self):
        """Non dimensional constant describing ratio between
        elastic and viscousc stresses."""
        return self.A**(1/self.n) * self.μ * self.τ**(1/self.n)

    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture stresses."""
        return self.μ * self.L / self.Gc


    def set_C1_to_one(self):
        """change L such that C1 = 1."""
        self.L = self.μ / (self.ρw * self.g)


    def set_C2_to_one(self):
        """change τ such that C2 = 1."""
        self.τ = (self.A**(1/self.n) * self.μ)**-self.n


    def set_C3l_to_one(self):
        """change regularisation length so C3*l=1."""
        self.l = 1/self.C3

    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.l = factor*h.max()*self.L
        

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.τ
    
    def nondimt2yrs(self,t):
        return t*self.τ/secperyr
    


class Material_with_uc:
    def __init__(self):

        # Default material properties
        self.ρi = 900 # Density of ice
        self.ρw = 1000 # Density of water
        self.g = 9.81 # Gravitational acceleration
        self.E = 9.33e9 # Young's modulus
        self.ν = 0.325 # Poisson's ratio
        self.A = 1.2e-25 # Flow law parameter
        self.n = 3.0 # Flow law exponent
        self.Gc = 1.0 # Fracture toughness
        self.L = 100 # Characteristic length
        self.l = 0.5 # Regularisation length
        self.slope_angle = 0.0
        self.σc =  0.1e6 # tensile strength
        self.ψcrit = 1.0 
        self.uc = 1e-2 # Critical displacement
        self.τ = secperyr # Characteristic time in seconds
        self.patm = 1e5 # Atmospheric pressure

        # self.lstar = l/L # Regularisation length
        # self.Hc = self.μ*(uc/L)**2
        # self.ψcritstar = self.ψcrit / self.Hc
        # self.pc = self.μ * uc / L
        # self.pwc = self.ρw * self.g * L
        # self.ρratio = self.ρi/self.ρw

    @property
    def q(self):
        """Calculate the paramerter q for Lo et al. degradation model."""
        a = 3*self.Gc*self.E / (8*self.l*self.σc**2)
        q = 1.0
        for i in range(100):
            q = a/np.log((1+q)/q)
        return q
        


    @property
    def uc_star(self):
        return self.uc/self.L

    
    @property
    def Hc(self):
        return self.μ*(self.uc/self.L)**2
    
    @property
    def ψcritstar(self):
        return self.ψcrit / self.Hc
    
    @property
    def pc(self):
        return self.μ * self.uc / self.L
    
    @property
    def pwc(self):
        return self.ρw * self.g * self.L
    
    @property
    def ρratio(self):
        return self.ρi/self.ρw



    @property
    def lstar(self):
        return self.l/self.L


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
        return self.L**2 * self.ρw * self.g / (self.uc * self.μ)

    @property
    def C2(self):
        """Non dimensional constant describing ratio between
        elastic and viscousc stresses."""
        return self.A**(1/self.n) * (self.uc/self.L)**(1-1/self.n) * \
                self.μ * self.τ**(1/self.n)

    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture stresses."""
        return self.μ * self.uc**2 / (self.Gc * self.L)


    def set_C1_to_one(self):
        """change uc such that C1 = 1."""
        self.uc = self.L**2 * self.ρw * self.g /  self.μ


    def set_C2_to_one(self):
        """change τ such that C2 = 1."""
        self.τ = (self.A**(1/self.n) * (self.uc/self.L)**(1-1/self.n) * \
                self.μ)**-self.n


    def set_C3l_to_one(self):
        """change regularisation length so C3*l=1."""
        self.l = 1/self.C3

    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.lstar = factor*h.max()
        

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.τ
    
    def nondimt2yrs(self,t):
        return t*self.τ/secperyr






    



