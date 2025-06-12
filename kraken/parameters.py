from kraken.utilities import mesh_sizes
import numpy as np
secperyr = 31556926



class Params_total_velocity:
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
        self.dt = 60*60*24 # Time step in seconds
        self.patm = 1e5 # Atmospheric pressure

    @property
    def λ(self):
        """Calculate the first Lamé parameter."""
        return self.ν * self.E / ((1 + self.ν) * (1 - 2 * self.ν))

    @property
    def μ(self):
        """Calculate the shear modulus (second Lamé parameter)."""
        return self.E / (2 * (1 + self.ν))

    @property
    def q(self):
        """Calculate the paramerter q for Lo et al. degradation model."""
        a = 3*self.Gc*self.E / (8*self.l*self.σc**2)
        q = 1.0
        for i in range(100):
            q = a/np.log((1+q)/q)
        return q
    
    @property
    def γdot(self):
        """Calculate the characteristic strain rate."""
        return self.uc / (self.L* self.dt)
    
    @property
    def ηc(self):
        """Calculate the characteristic viscosity."""
        return self.A**(-1/self.n) * self.γdot**((1-self.n)/self.n)
    
    
    @property
    def τ(self):
        """Relaxation time."""
        return self.ηc/ self.μ
    

    @property
    def De(self):
        """Deborah number."""
        return self.τ / self.dt
    
    @property
    def uc(self):
        return (self.ρc * self.g)**self.n * self.A * self.dt * self.L**(self.n+1)
    
    
    @property
    def ucstar(self):
        """Non dimensional critical displacement."""
        return self.uc/self.L
    
    @property
    def ue(self):
        return self.uc * self.De
    
    @property
    def uestar(self):

        return self.ue / self.dt
    
    
    @property
    def Hc(self):
        return self.μ*(self.De*self.ucstar)**2
    
    @property
    def ψcritstar(self):
        return self.ψcrit / self.Hc
    
    @property
    def pc(self):
        return self.ηc * self.uc / (self.L * self.dt)
    
    @property
    def pwc(self):
        return self.ρc * self.g * self.L
    
    @property
    def ρc(self):
        """Characteristic density."""
        return self.ρw
    
    @property
    def ρistar(self):
        return self.ρi/self.ρc
    
    @property
    def patmstar(self):
        return self.patm / self.pwc
    
    @property
    def δ(self):
        """Non dimensional density difference."""
        return 1.0 - self.ρistar

    @property
    def lstar(self):
        return self.l/self.L

   
    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture stresses."""
        return self.Hc * self.L / self.Gc


    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.l = factor*h.max()*self.L
        






    






class Params_no_uc:
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

    def set_C1C2_to_one(self):

        self.τ = (self.ρw * self.g * self.L * self.A**(1/self.n))**-self.n

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
    


class Params_with_uc:
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
        self.dt = secperyr # Characteristic time in seconds
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
    def uc(self):
        return self.ρc * self.g * self.L**2 / self.μ

    @property
    def ucstar(self):
        return self.uc/self.L
    
    @property
    def τ(self):
        """Relaxation time."""
        return self.A**(-1) * self.ucstar**(1-self.n) * self.μ**(-self.n)
    

    @property
    def γdot(self):
        """Calculate the characteristic strain rate."""
        return self.ucstar / self.τ
    
    @property
    def ηc(self):
        """Calculate the characteristic viscosity."""
        return self.A**(-1/self.n) * self.γdot**((1-self.n)/self.n)
    

    @property
    def dtstar(self):
        """Non dimensional time step."""
        return self.dt / self.τ
    
    @property
    def De(self):
        """Deborah number."""
        return 1 / self.dtstar
    
    @property
    def Hc(self):
        return self.μ*(self.ucstar)**2
    
    @property
    def ψcritstar(self):
        return self.ψcrit / self.Hc
    
    @property
    def pc(self):
        return self.μ * self.ucstar
    
    @property
    def ρc(self):
        """Characteristic density."""
        return self.ρw
    
    @property
    def pwc(self):
        return self.ρc * self.g * self.L
    
    @property
    def ρistar(self):
        return self.ρi/self.ρc
    
    
    
    @property
    def δ(self):
        """Non dimensional density difference."""
        return 1.0 - self.ρistar
    
    @property
    def patmstar(self):
        return self.patm / self.pwc



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



    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.lstar = factor*h.max()
        

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.dt
    
    def nondimt2yrs(self,t):
        return t*self.dt/secperyr






    



