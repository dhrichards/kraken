''' Some general purpose integrators for time stepping
    rhs is a lambda function that takes the current state
    and returns the right hand side of the ODE, e.g.
    rhs = lambda f: ufl.dot(M,f) for fabric'''

def ForwardEuler(rhs,f,dt):
    return f + dt*rhs(f)

def RK4(rhs,f,dt):
    k1 = rhs(f)
    k2 = rhs(f + 0.5*dt*k1)
    k3 = rhs(f + 0.5*dt*k2)
    k4 = rhs(f + dt*k3)

    return f + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def SSPRK43(rhs,f,dt):
    k1 = rhs(f)
    k2 = rhs(f + 0.5*dt*k1)
    k3 = rhs(f + 0.5*dt*k1 + 0.5*dt*k2)
    k4 = rhs(f + dt*k1/6 + dt*k2/6 + dt*k3/6)

    return f + dt*(k1/6 + k2/6 + k3/6 + k4/2)


def SSPRK104(rhs,f,dt):
    k1 = rhs( f)
    k2 = rhs( f + dt*k1/6)
    k3 = rhs( f + dt*k1/6 + dt*k2/6)
    k4 = rhs( f + dt*k1/6 + dt*k2/6 + dt*k3/6)
    k5 = rhs( f + dt*k1/6 + dt*k2/6 + dt*k3/6 + dt*k4/6)
    k6 = rhs( f + dt*k1/15 + dt*k2/15 + dt*k3/15 + dt*k4/15 + dt*k5/15)
    k7 = rhs( f + dt*k1/15 + dt*k2/15 + dt*k3/15 + dt*k4/15 + dt*k5/15 + dt*k6/6)
    k8 = rhs( f + dt*k1/15 + dt*k2/15 + dt*k3/15 + dt*k4/15 + dt*k5/15 + dt*k6/6 + dt*k7/6)
    k9 = rhs( f + dt*k1/15 + dt*k2/15 + dt*k3/15 + dt*k4/15 + dt*k5/15 + dt*k6/6 + dt*k7/6 + dt*k8/6)
    k10 =rhs( f + dt*k1/15 + dt*k2/15 + dt*k3/15 + dt*k4/15 + dt*k5/15 + dt*k6/6 + dt*k7/6 + dt*k8/6 + dt*k9/6)

    return f + 0.1*dt*(k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 + k10)