from dolfinx import fem, default_scalar_type
import numpy as np
import ufl
from kraken.utilities import mesh_sizes
secperyr = 31556926




class Params_with_uc:
    def __init__(self,msh):

        self.msh = msh

        # Default material properties
        self.Ttop = fem.Constant(msh,default_scalar_type(-20.0)) # Temperature in Celsius
        self.Tbot = fem.Constant(msh,default_scalar_type(-20.0)) # Temperature in Celsius
        self.ρi = fem.Constant(msh,default_scalar_type(900)) # Density of ice
        self.ρw = fem.Constant(msh,default_scalar_type(1000)) # Density of water
        self.g = fem.Constant(msh,default_scalar_type(9.81)) # Gravitational acceleration
        self.E = fem.Constant(msh,default_scalar_type(9.33e9)) # Young's modulus
        self.ν = fem.Constant(msh,default_scalar_type(0.325)) # Poisson's ratio
        self.A0 = fem.Constant(msh,default_scalar_type(1.2e-25)) # Flow law parameter
        self.n = fem.Constant(msh,default_scalar_type(3.0)) # Flow law exponent
        self.H = fem.Constant(msh,default_scalar_type(100)) # Characteristic length
        self.l = fem.Constant(msh,default_scalar_type(0.5)) # Regularisation length
        self.dt = fem.Constant(msh,default_scalar_type(secperyr)) # Characteristic time in seconds
        self.patm = fem.Constant(msh,default_scalar_type(0.0)) # Atmospheric pressure
        self.ge_tol = fem.Constant(msh,default_scalar_type(1e-12)) # Elastic degradation tolerance
        self.crack_level_above_sea = fem.Constant(msh,default_scalar_type(0.0)) # Water level for hydrostatic pressure
        self.sea_level = fem.Constant(msh,default_scalar_type(0.9*self.H.value)) # Sea level height
        self.length = fem.Constant(msh,default_scalar_type(16e3)) # Length of domain in flow direction

        self.σt0 = fem.Constant(msh,default_scalar_type(0.2e6))
        self.σt_deg = fem.Constant(msh,default_scalar_type(0.04e6))

        self.Kic = fem.Constant(msh,default_scalar_type(100e3)) # Fracture toughness
        
        self.cp = fem.Constant(msh,default_scalar_type(2100)) # Specific heat capacity
        self.κ = fem.Constant(msh,default_scalar_type(2)) # Thermal conductivity

        self.friction_angle = fem.Constant(msh, default_scalar_type(np.atan(0.3))) # Friction angle in radians
        self.cohesion = fem.Constant(msh, default_scalar_type(164e3))


    @property
    def σc(self):
        return self.σt0 - self.σt_deg*(self.T)
    
    @property
    def T(self):
        x = ufl.SpatialCoordinate(self.msh)
        z = x[self.msh.geometry.dim-1]
        return self.Tbot + (self.Ttop - self.Tbot)*z

    @property
    def ψcrit(self):
        return self.σc**2 / (2*self.E)

    @property
    def Gc(self):
        return self.Kic**2*(1-self.ν**2)/self.E


    
    @property
    def B(self):
        return 2*ufl.sin(self.friction_angle)/(np.sqrt(3)*(3+ufl.sin(self.friction_angle)))

 
    @property
    def cohesion_star(self):
        return self.cohesion / self.μ

    @property
    def uc(self):
        return self.ρc * self.g * self.H**2 / self.μ

    @property
    def ucstar(self):
        return self.uc/self.H
    
    @property
    def ucstar_float(self):
        μ = self.E.value / (2 * (1 + self.ν.value))
        return self.ρc.value * self.g.value * self.H.value / μ
    
    @property
    def crack_level_star(self):
        return self.crack_level_above_sea / self.H + self.sea_level_star
    
    @property
    def sea_level_star(self):
        return self.sea_level / self.H
    
    @property
    def τ(self):
        """Relaxation time."""
        return self.A0**(-1) * self.ucstar**(1-self.n) * self.μ**(-self.n)
    

    @property
    def γdot(self):
        """Calculate the characteristic strain rate."""
        return self.ucstar / self.τ
    
    @property
    def ηc(self):
        """Calculate the characteristic viscosity."""
        return self.A0**(-1/self.n) * self.γdot**((1-self.n)/self.n)
    

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
        return self.ρi
    
    @property
    def pwc(self):
        return self.ρc * self.g * self.H
    
    @property
    def ρistar(self):
        return self.ρi/self.ρc
    
    @property
    def ρwstar(self):
        return self.ρw/self.ρc
    
    @property
    def δ(self):
        return 1-self.ρi/self.ρw
    
    # @property
    # def σc(self):
    #     return self.μ*self.ucstar
   
    
    @property
    def patmstar(self):
        return self.patm / self.pwc



    @property
    def lstar(self):
        return self.l/self.H


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
        return self.A0**(1/self.n) * (self.uc/self.H)**(1-1/self.n) * \
                self.μ * self.τ**(1/self.n)

    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture stresses."""
        return self.μ * self.uc**2 / (self.Gc * self.H)

    @property
    def C_inertia(self):
        return self.ρc * self.H**2 / (self.μ * self.τ**2)
    
    @property
    def C_temperature(self):
        return self.ηc*(self.ucstar/self.τ)**2*self.τ/(self.ρc*self.cp)
    
    @property
    def κstar(self):
        return self.κ*self.τ/(self.ρc*self.cp*self.H**2)

    @property
    def length_star(self):
        return self.length / self.H

    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.lstar = factor*h.max()
        

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.dt
    
    def nondimt2yrs(self,t):
        return t*self.dt/secperyr






    


