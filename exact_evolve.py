import numpy as np

class Parameters:
    """
    This class contains the parameters necessary for evolution
    """
    def __init__(self, a, F0, field, t, U):
        self.a = a
        self.F0 = F0
        self.field = field
        self.freq = field / (2 * np.pi)
        self.t = t
        self.U = U

def phi_tl(current_time, lat, cycles):
    """
    Calculates phi
    :param current_time: time in the evolution
    :return: phi
    """
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return phi

def evolve_psi(current_time, psi, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    """
    Evolves psi
    :param current_time: time in evolution
    :param psi: the current wavefunction
    :param phi_func: the function used to calculate phi
    :return: -i * H|psi>
    """

    # print("Simulation Progress: |" + "#" * int(current_time * lat.freq) + " " * (10 - int(current_time * lat.freq))
    #       + "|" + "{:.2f}".format(current_time * lat.freq * 10) + "%", end="\r")

    phi = phi_func(current_time, lat, cycles)

    a = -1j * (-lat.t * (np.exp(-1j*phi)*hop_left.static.dot(psi) + np.exp(1j*phi)*hop_right.static.dot(psi))
               + lat.U * onsite.static.dot(psi))

    return a

def H_expec(psis, times, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    """
    Calculates expectation of the hamiltonian
    :param psis: list of states at every point in the time evolution
    :param times: the times at which psi was calculated
    :param phi_func: the function used to calculate phi
    :return: an array of the expectation values of a Hamiltonian
    """
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        phi = phi_func(current_time, lat, cycles)
        # H|psi>
        Hpsi = -lat.t * (np.exp(-1j*phi) * hop_left.dot(psi) + np.exp(1j*phi) * hop_right.dot(psi)) + \
            lat.U * onsite.dot(psi)
        # <psi|H|psi>
        expec.append((np.vdot(psi, Hpsi)).real)
    return np.array(expec)

def J_expec(psis, times, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    """
    Calculates expectation of the current density
    :param psis: list of states at every point in the time evolution
    :param times: the times at which psi was calculated
    :param phi_func: the function used to calculate phi
    :return: an array of the expectation values of a density
    """
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        phi = phi_func(current_time, lat, cycles)
        # J|psi>
        Jpsi = -1j*lat.a*lat.t* (np.exp(-1j*phi) * hop_left.dot(psi) - np.exp(1j*phi) * hop_right.dot(psi))
        # <psi|J|psi>
        expec.append((np.vdot(psi, Jpsi)).real)
    return np.array(expec)
