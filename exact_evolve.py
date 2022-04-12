import numpy as np
from tools import phi_tl

def evolve_psi(current_time, psi, onsite, hop_left, hop_right, lat, phi_func=phi_tl):
    """
    Evolves psi
    :param current_time: time in evolution
    :param psi: the current wavefunction
    :param phi_func: the function used to calculate phi
    :return: -i * H|psi>
    """
    freq = 2 * np.pi * lat.field

    phi = phi_func(current_time, lat)

    a = -1j * (-lat.t0 * (np.exp(-1j*phi)*hop_left.static.dot(psi) + np.exp(1j*phi)*hop_right.static.dot(psi))
               + lat.u * onsite.static.dot(psi))

    return a

def H_expec(psis, times, onsite, hop_left, hop_right, lat, phi_func=phi_tl):
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
        phi = phi_func(current_time, lat)
        # H|psi>
        Hpsi = -lat.t0 * (np.exp(-1j*phi) * hop_left.dot(psi) + np.exp(1j*phi) * hop_right.dot(psi)) + \
            lat.u * onsite.dot(psi)
        # <psi|H|psi>
        expec.append((np.vdot(psi, Hpsi)).real)
    return np.array(expec)

def J_expec(psis, times, hop_left, hop_right, lat, phi_func=phi_tl):
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
        phi = phi_func(current_time, lat)
        # J|psi>
        Jpsi = -1j*lat.a*lat.t0* (np.exp(-1j*phi) * hop_left.dot(psi) - np.exp(1j*phi) * hop_right.dot(psi))
        # <psi|J|psi>
        expec.append((np.vdot(psi, Jpsi)).real)
    return np.array(expec)
