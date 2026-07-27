from dolfinx import fem, default_scalar_type
import numpy as np
import ufl
from kraken.utilities import mesh_sizes
secperyr = 31556926




class Params:
    """
    Container for physical and numerical parameters used by Kraken.

    This class stores dimensional material properties, dimensionless constants,
    and numerical parameters as DOLFINx ``Constant`` objects, so that they can be updated during a simulation.
    The class is created and then a value can be updated as e.g.:
    params.T.value = -20
    They can also be overloaded as functions of space using ufl.SpatialCoordinate
    Dervied quanitites auto-update as properties

    Inputs
    ----------
    msh : dolfinx.mesh.Mesh
        Mesh on which the DOLFINx constants are defined.

    Attributes
    ----------
    T : Ice temperature (degrees Celsius).

    ρi : Ice density (kg m⁻³).

    ρw : Water density (kg m⁻³).

    g : Gravitational acceleration (m s⁻²).

    E : Young's modulus (Pa).

    ν : Poisson's ratio.

    A0 : Glen flow law parameter.

    n : Glen flow law exponent.

    H : Characteristic length scale (m).

    l : Phase-field regularisation length scale (m).

    dt : Timestep (s).

    patm: Atmoshpheric pressure (Pa)

    ge_tol: Tolerance in degradation function 

    crack_level_above_sea : Water level in crack above sea level (default 0)

    viscosity_tol: tolerance in viscosity degradation η → (1-d)²η + viscosity_tol 

    σt : Tensile strength (Pa).

    Kic : Fracture toughness (Pa m¹ᐟ²).
    """
    def __init__(self,msh):

        self.msh = msh

        # Default material properties
        self.T = fem.Constant(msh,default_scalar_type(-20.0)) # Temperature in Celsius
        self.ρi = fem.Constant(msh,default_scalar_type(900)) # Density of ice
        self.ρw = fem.Constant(msh,default_scalar_type(1000)) # Density of water
        self.g = fem.Constant(msh,default_scalar_type(9.81)) # Gravitational acceleration
        self.E = fem.Constant(msh,default_scalar_type(9.33e9)) # Young's modulus
        self.ν = fem.Constant(msh,default_scalar_type(0.325)) # Poisson's ratio
        self.A0 = fem.Constant(msh,default_scalar_type(1.2e-25)) # Flow law parameter
        self.n = fem.Constant(msh,default_scalar_type(3.0)) # Flow law exponent
        self.H = fem.Constant(msh,default_scalar_type(100)) # Characteristic length
        self.l = fem.Constant(msh,default_scalar_type(0.5)) # Regularisation length
        self.dt = fem.Constant(msh,default_scalar_type(3600*24)) # Timestep in seconds
        self.patm = fem.Constant(msh,default_scalar_type(0.0)) # Atmospheric pressure
        self.ge_tol = fem.Constant(msh,default_scalar_type(1e-12)) # Elastic degradation tolerance
        self.crack_level_above_sea = fem.Constant(msh,default_scalar_type(0.0)) # Water level for hydrostatic pressure
        self.viscosity_tol = fem.Constant(msh,default_scalar_type(0.1)) # Viscosity regularisation

        self.σt = fem.Constant(msh,default_scalar_type(0.2e6)) # Tensile strength
        self.Kic = fem.Constant(msh,default_scalar_type(100e3)) # Fracture toughness

        self._sea_level_override = None
        self._ρc_override = None

    @property
    def sea_level(self):
        ''' Sea level, can be overwritten for non-flotation'''
        if self._sea_level_override is None:
            return self.ρi / self.ρw * self.H
        else:
            return self._sea_level_override

    @sea_level.setter
    def sea_level(self, value):
        self._sea_level_override = fem.Constant(self.msh,default_scalar_type(value))

    @property
    def ψcrit(self):
        '''Critical energy threshold'''
        return self.σt**2 / (2*self.E)

    @property
    def Gc(self):
        '''Critical energy release rate'''
        return self.Kic**2*(1-self.ν**2)/self.E


    @property
    def uc(self):
        '''Characteristic displacement'''
        return self.ρc * self.g * self.H**2 / self.μ

    @property
    def ucstar(self):
        '''Non-dimensional characteristic displacement'''
        return self.uc/self.H
    
    @property
    def ucstar_float(self):
        '''Non-dimensional characteristic displacement as float'''
        μ = self.E.value / (2 * (1 + self.ν.value))
        return self.ρc.value * self.g.value * self.H.value / μ
    
    @property
    def crack_level_star(self):
        '''Non dimensional crack water level'''
        return self.crack_level_above_sea / self.H + self.sea_level_star
    
    @property
    def sea_level_star(self):
        '''Non dimensional sea-level'''
        return self.sea_level / self.H
    
    @property
    def τ(self):
        '''Characteristic time'''
        return self.A0**(-1) * self.ucstar**(1-self.n) * self.μ**(-self.n)
    

    @property
    def γdot(self):
        '''Characteristic strain rate'''
        return self.ucstar / self.τ
    
    @property
    def ηc(self):
        '''Characteristic non-linear viscosity'''
        return self.A0**(-1/self.n) * self.γdot**((1-self.n)/self.n)
    

    @property
    def dtstar(self):
        '''Non-dimensional timestep'''
        return self.dt / self.τ
    
    @property
    def Hc(self):
        return self.μ*(self.ucstar)**2
    
    @property
    def ψcritstar(self):
        '''Non-dimensional energy threshold'''
        return self.ψcrit / self.Hc
    
    @property
    def pc(self):
        return self.μ * self.ucstar
    
    @property
    def ρc(self):
        if self._ρc_override == None:
            return self.ρi
        else:
            return self._ρc_override

    @ρc.setter
    def ρc(self, value):
        self._ρc_override = fem.Constant(self.msh,default_scalar_type(value))
    
    @property
    def pwc(self):
        return self.ρc * self.g * self.H
    
    @property
    def ρistar(self):
        '''Non-dimensional ice density'''
        return self.ρi/self.ρc
    
    @property
    def ρwstar(self):
        '''Non-dimensional water density'''
        return self.ρw/self.ρc
    
    @property
    def δ(self):
        return 1-self.ρi/self.ρw
    
    
    @property
    def patmstar(self):
        '''Non-dimensional atmospheric pressure'''
        return self.patm / self.pwc



    @property
    def lstar(self):
        '''Non-dimensional regularisation length'''
        return self.l/self.H


    @property
    def λ(self):
        """First Lamé parameter."""
        return self.ν * self.E / ((1 + self.ν) * (1 - 2 * self.ν))

    @property
    def μ(self):
        """Second Lamé parameter"""
        return self.E / (2 * (1 + self.ν))



    @property
    def C3(self):
        """Non dimensional constant describing ratio between
        elastic and fracture energies."""
        return self.μ * self.uc**2 / (self.Gc * self.H)
    

    def yrs2nondimt(self,yr):
        return yr*secperyr/self.dt
    
    def nondimt2yrs(self,t):
        return t*self.dt/secperyr






    


