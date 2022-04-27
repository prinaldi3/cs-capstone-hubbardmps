import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import Config
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfFermionSite

class Parameters:
    """
    Scales parameters to atomic units in terms of t_0.
    input units: eV (t, U)
    """

    def __init__(self, nsites, u, t, a, cycles, field, strength, pbc):
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

        self.pbc = pbc #periodic boundary conditions

class FHHamiltonian(FermiHubbardChain):
    def __init__(self, p, phi):
        t0 = p.t0 * np.exp(-1j * phi)
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", "explicit_plus_hc":True,
        "L":p.nsites, "mu":0, "V":0, "U":p.u, "t":t0}
        model_params = Config(model_dict, "FHHam-U{}".format(p.u))
        FermiHubbardChain.__init__(self, model_params)

"""
This is the tracking hamiltonian instantiated by passing in a current expectation
that we would like to track
"""
class TrackingHamiltonian(FermiHubbardChain):
    """
    Parameters:
        p - an instance of the parameters class
        tcurrent - the value of the current we would like to track at some time
        tebd - an instance of the Engine class
    """
    def __init__(self, p, tcurrent, tebd):
        expec = tebd.nnop.H_MPO.expectation_value(tebd.psi)
        rpsi = np.abs(expec)
        thetapsi = np.angle(expec)
        x = tcurrent.real / (2 * p.a * p.t0 * rpsi)
        pplus = -p.t0 * (np.sqrt(1 - x**2) + 1j * x)
        t0 = pplus * np.exp(-1j * thetapsi)
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", "explicit_plus_hc":True,
        "L":p.nsites, "mu":0, "V":0, "U":p.u, "t":t0}
        model_params = Config(model_dict, "FHHam-U{}".format(p.u))
        FermiHubbardChain.__init__(self, model_params)

class FHCurrentModel(CouplingMPOModel):
    def __init__(self, p, phi):
        t0 = p.t0 * np.exp(-1j * phi)
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
        "L":p.nsites, "t":t0, "a":p.a}
        model_params = Config(model_dict, "FHCurrent-U{}".format(p.u))
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = model_params.get('t', 1.)
        a = model_params.get('a', 4 * t * 1.889726125 * 0.036749323)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # the -dx is necessary for hermitian conjugation, see documentation
            self.add_coupling(-1j * a * t, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(1j * a * np.conjugate(t), u2, 'Cdu', u1, 'Cu', -dx)
            self.add_coupling(-1j * a * t, u1, 'Cdd', u2, 'Cd', dx)
            self.add_coupling(1j * a * np.conjugate(t), u2, 'Cdd', u1, 'Cd', -dx)

class FHCurrent(FHCurrentModel, NearestNeighborModel):
    default_lattice = Chain
    force_default_lattice = True

class FHNearestNeighborModel(CouplingMPOModel):
    def __init__(self, p):
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
        "L":p.nsites}
        model_params = Config(model_dict, "FHNearestNeighbors")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # the -dx is necessary for hermitian conjugation, see documentation
            self.add_coupling(1, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(1, u1, 'Cdd', u2, 'Cd', dx)

class FHNearestNeighbor(FHNearestNeighborModel, NearestNeighborModel):
    default_lattice = Chain
    force_default_lattice = True

def phi_tl(time, p):
    """
    Calculate transform limited phi at time
    Params:
        p - an instance of Parameters
    """
    return (p.a * p.strength / p.field) * (np.sin(p.field * time / (2*p.cycles))**2) * np.sin(p.field * time)

def phi_tracking(time, p, target_current, tebd):
    """
    Calculates phi(time) for some current expectation we would like to track
    Params:
        p - an instance of Parameters
        target_current - the value of the current we are tracking at time
        tebd - an instance of the Engine class
    """
    expec = tebd.nnop.H_MPO.expectation_value(tebd.psi)
    r = np.abs(expec)
    theta = np.angle(expec)
    return np.arcsin( (-target_current) / (2*p.a*p.t0*r) ) + theta

def relative_error(exact, mps):
    return 100 * np.linalg.norm(exact - mps) / np.linalg.norm(exact)
