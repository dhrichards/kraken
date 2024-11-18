

secperyr = 60*60*24*365.25

class MaterialProperties:
    def __init__(self, ρi=900, ρw=1000, g=9.81, E=9.33e9, ν=0.325, A=1.2e-25, n=3.0, Gc=1.0, ψcrit=1.0, l=5e-3, uc=1e-2, L=100.0, τ=secperyr):
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
        self.l = l # Regularisation length
        self.uc = uc # Critical displacement
        self.L = L # Characteristic length
        self.τ = τ # Characteristic time in seconds




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
    



