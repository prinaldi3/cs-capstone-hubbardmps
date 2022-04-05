import numpy as np
class Parameters:
    """
    Scales parameters to atomic units in terms of t_0.
    input units: eV (t, U)
    """

    def __init__(self, nsites, u, t, a, cycles, field, strength):
        self.nsites = nsites
        self.nup = nsites // 2 + nsites % 2
        self.ndown = nsites // 2

        self.u = u / t
        self.t0 = 1.

        self.cycles = cycles

        # CONVERTING TO ATOMIC UNITS, w/ energy normalized to t_0
        factor = 1 / (t * 0.036749323)
        self.field = field * factor * 0.0001519828442
        self.a = a * 1.889726125/factor
        self.strength = strength * 1.944689151e-4 * (factor**2)

def phi_tl(time, p):
    """
    Calculate transform limited phi at time
    Params:
        p - an instance of Parameters
    """
    return (p.a * p.strength / p.field) * (np.sin(p.field * time / (2*p.cycles))**2) * np.sin(p.field * time)
