from dolfinx import fem, default_scalar_type
from kraken.utilities import mesh_sizes
secperyr = 31556926




class Params_with_uc:
    def __init__(self,msh):

        # Default material properties
        self.T = fem.Constant(msh,default_scalar_type(-20.0)) # Temperature in Celsius

        self.ρi = fem.Constant(msh,default_scalar_type(900)) # Density of ice
        self.ρw = fem.Constant(msh,default_scalar_type(1000)) # Density of water
        self.g = fem.Constant(msh,default_scalar_type(9.81)) # Gravitational acceleration
        self.E = fem.Constant(msh,default_scalar_type(9.33e9)) # Young's modulus
        self.ν = fem.Constant(msh,default_scalar_type(0.325)) # Poisson's ratio
        self.A0 = fem.Constant(msh,default_scalar_type(1.2e-25)) # Flow law parameter
        self.n = fem.Constant(msh,default_scalar_type(3.0)) # Flow law exponent
        self.Gc = fem.Constant(msh,default_scalar_type(1.0)) # Fracture toughness
        self.L = fem.Constant(msh,default_scalar_type(100)) # Characteristic length
        self.l = fem.Constant(msh,default_scalar_type(0.5)) # Regularisation length
        self.σcrit = fem.Constant(msh,default_scalar_type(0.2e6)) # tensile strength
        self.ψcrit = fem.Constant(msh,default_scalar_type(1.0)) 
        self.dt = fem.Constant(msh,default_scalar_type(secperyr)) # Characteristic time in seconds
        self.patm = fem.Constant(msh,default_scalar_type(1e5)) # Atmospheric pressure
        self.gv_tol = fem.Constant(msh,default_scalar_type(1e-5)) # Viscous degradation tolerance
        


    # @property
    # def q(self):
    #     """Calculate the paramerter q for Lo et al. degradation model."""
    #     a = 3*self.Gc*self.E / (8*self.l*self.σcrit**2)
    #     q = 1.0
    #     for i in range(100):
    #         q = a/np.log((1+q)/q)
    #     return q
        

    @property
    def uc(self):
        return self.ρc * self.g * self.L**2 / self.μ

    @property
    def ucstar(self):
        return self.uc/self.L
    
    @property
    def ucstar_float(self):
        μ = self.E.value / (2 * (1 + self.ν.value))
        return self.ρc.value * self.g.value * self.L.value / μ
    
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
        return self.ρw
    
    @property
    def pwc(self):
        return self.ρc * self.g * self.L
    
    @property
    def ρistar(self):
        return self.ρi/self.ρc
    
    @property
    def σc(self):
        return self.μ*self.ucstar
    
    # @property
    # def σcritstar(self):
    #     """Non dimensional tensile strength."""
    #     return self.σcrit / self.σc
    
    
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
        return self.A0**(1/self.n) * (self.uc/self.L)**(1-1/self.n) * \
                self.μ * self.τ**(1/self.n)

    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture stresses."""
        return self.μ * self.uc**2 / (self.Gc * self.L)

    @property
    def C_inertia(self):
        return self.ρc * self.L**2 / (self.μ * self.τ**2)


    def set_Gc_AT1_lowerorder(self):
        self.Gc.value = 8* self.σcrit.value**2 * self.l.value / (3*self.E.value)

    def set_Gc_AT1_higherorder(self):
        self.Gc.value = 2* self.σcrit.value**2 * self.l.value / self.E.value



    def set_l_from_mesh(self,msh,factor=2):
        """Set the regularisation length from the mesh."""
        h = mesh_sizes(msh)
        self.lstar = factor*h.max()
        

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.dt
    
    def nondimt2yrs(self,t):
        return t*self.dt/secperyr






    


