r"""Time evolving block decimation (TEBD).

The TEBD algorithm (proposed in :cite:`vidal2004`) uses a trotter decomposition of the
Hamiltonian to perform a time evoltion of an MPS. It works only for nearest-neighbor hamiltonians
(in tenpy given by a :class:`~tenpy.models.model.NearestNeighborModel`),
which can be written as :math:`H = H^{even} + H^{odd}`,  such that :math:`H^{even}` contains the
the terms on even bonds (and similar :math:`H^{odd}` the terms on odd bonds).
In the simplest case, we apply first :math:`U=\exp(-i*dt*H^{even})`,
then :math:`U=\exp(-i*dt*H^{odd})` for each time step :math:`dt`.
This is correct up to errors of :math:`O(dt^2)`, but to evolve until a time :math:`T`, we need
:math:`T/dt` steps, so in total it is only correct up to error of :math:`O(T*dt)`.
Similarly, there are higher order schemata (in dt) (for more details see :meth:`Engine.update`).

Remember, that bond `i` is between sites `(i-1, i)`, so for a finite MPS it looks like::

    |     - B0 - B1 - B2 - B3 - B4 - B5 - B6 -
    |       |    |    |    |    |    |    |
    |       |----|    |----|    |----|    |
    |       | U1 |    | U3 |    | U5 |    |
    |       |----|    |----|    |----|    |
    |       |    |----|    |----|    |----|
    |       |    | U2 |    | U4 |    | U6 |
    |       |    |----|    |----|    |----|
    |                   .
    |                   .
    |                   .

After each application of a `Ui`, the MPS needs to be truncated - otherwise the bond dimension
`chi` would grow indefinitely. A bound for the error introduced by the truncation is returned.

If one chooses imaginary :math:`dt`, the exponential projects
(for sufficiently long 'time' evolution) onto the ground state of the Hamiltonian.

.. note ::
    The application of DMRG is typically much more efficient than imaginary TEBD!
    Yet, imaginary TEBD might be usefull for cross-checks and testing.

"""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import datetime

from tenpy.linalg import np_conserved as npc
from tenpy.algorithms.truncation import svd_theta, TruncationError
from tenpy.tools.params import asConfig, Config
from tenpy.linalg.random_matrix import CUE

from tools import FHHamiltonian, FHCurrent, FHNearestNeighbor, phi_tl

__all__ = ['Engine', 'RandomUnitaryEvolution']


class Engine:
    """Time Evolving Block Decimation (TEBD) algorithm.

    .. deprecated :: 0.6.0
        Renamed parameter/attribute `TEBD_params` to :attr:`options`.


    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.model.NearestNeighborModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    p : :class:`tools.Parameters`
        An instance of the Parameters class
    phi_func : function
        The function that calculates phi
    options : dict
        Further optional parameters as described in the tables in
        :func:`run` and :func:`run_GS` for more details.
        Use ``verbose=1`` to print the used parameters during runtime.

    Options
    -------
    .. cfg:config :: TEBD

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncate`.
        start_time : float
            Initial value for :attr:`evolved_time`.
        start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`.

    Attributes
    ----------
    verbose : int
        See :cfg:option:`TEBD.verbose`.
    options: :class:`~tenpy.tools.params.Config`
        Optional parameters, see :meth:`run` and :meth:`run_GS` for more details.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    time: float
        Indicates how long psi has been evolved (evolved_time only does this after all steps)
    trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
        The error of the represented state which is introduced due to the truncation during
        the sequence of update steps.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    model : :class:`~tenpy.models.model.NearestNeighborModel`
        The model defining the Hamiltonian.
    p : :class:`tools.Parameters`
        An instance of the Parameters class
    _U : list of list of :class:`~tenpy.linalg.np_conserved.Array`
        Exponentiated `H_bond` (bond Hamiltonians), i.e. roughly ``exp(-i H_bond dt_i)``.
        First list for different `dt_i` as necessary for the chosen `order`,
        second list for the `L` different bonds.
    _U_param : dict
        A dictionary containing the information of the latest created `_U`.
        We don't recalculate `_U` if those parameters didn't change.
    _trunc_err_bonds : list of :class:`~tenpy.algorithms.truncation.TruncationError`
        The *local* truncation error introduced at each bond, ignoring the errors at other bonds.
        The `i`-th entry is left of site `i`.
    _update_index : None | (int, int)
        The indices ``i_dt,i_bond`` of ``U_bond = self._U[i_dt][i_bond]`` during update_step.
    """
    def __init__(self, psi, model, p, phi_func, options):
        self.options = options = asConfig(options, "TEBD")
        self.trunc_params = options.subconfig('trunc_params')
        self.psi = psi
        self.model = model
        self.currentop = FHCurrent(p, 0)
        self.nnop = FHNearestNeighbor(p)  # nearest neighbor operator
        self.p = p
        self.phi_func = phi_func
        self.evolved_time = self.time = options.get('start_time', 0.)
        self.trunc_err = options.get('start_trunc_err', TruncationError())
        self._U = None
        self._U_param = {}
        self._trunc_err_bonds = [TruncationError() for i in range(psi.L + 1)]
        self._update_index = None

    @property
    def TEBD_params(self):
        warnings.warn("renamed self.TEBD_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    @property
    def trunc_err_bonds(self):
        """truncation error introduced on each non-trivial bond."""
        return self._trunc_err_bonds[self.psi.nontrivial_bonds]

    def run(self, tracking_current=None):
        """(Real-)time evolution with TEBD (time evolving block decimation).

        .. cfg:configoptions :: TEBD

            dt : float
                Time step.
            N_steps : int
                Number of time steps `dt` to evolve.
                The Trotter decompositions of order > 1 are slightly more efficient
                if more than one step is performed at once.
            order : int
                Order of the algorithm. The total error scales as ``O(t*dt^order)``.

        """
        # initialize parameters
        delta_t = self.options.get('dt', 0.1)
        N_steps = self.options.get('N_steps', 10)
        TrotterOrder = self.options.get('order', 2)

        self.calc_U(TrotterOrder, delta_t, type_evo='real', E_offset=None)

        trunc_err, times, energies, currents, phis = self.update(N_steps, delta_t, tracking_current)
        return times, energies, currents, phis

    @staticmethod
    def suzuki_trotter_time_steps(order):
        """Return time steps of U for the Suzuki Trotter decomposition of desired order.

        See :meth:`suzuki_trotter_decomposition` for details.

        Parameters
        ----------
        order : int
            The desired order of the Suzuki-Trotter decomposition.

        Returns
        -------
        time_steps : list of float
            We need ``U = exp(-i H_{even/odd} delta_t * dt)`` for the `dt` returned in this list.
        """
        if order == 1:
            return [1.]
        elif order == 2:
            return [0.5, 1.]
        elif order == 4:
            t1 = 1. / (4. - 4.**(1 / 3.))
            t3 = 1. - 4. * t1
            return [t1 / 2., t1, (t1 + t3) / 2., t3]
        elif order == '4_opt':
            # Eq (30a) of arXiv:1901.04974
            a1 = 0.095848502741203681182
            b1 = 0.42652466131587616168
            a2 = -0.078111158921637922695
            b2 = -0.12039526945509726545
            return [a1, b1, a2, b2, 0.5 - a1 - a2, 1. - 2 * (b1 + b2)]  # a1 b1 a2 b2 a3 b3
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    @staticmethod
    def suzuki_trotter_decomposition(order, N_steps):
        r"""Returns list of necessary steps for the suzuki trotter decomposition.

        We split the Hamiltonian as :math:`H = H_{even} + H_{odd} = H[0] + H[1]`.
        The Suzuki-Trotter decomposition is an approximation
        :math:`\exp(t H) \approx prod_{(j, k) \in ST} \exp(d[j] t H[k]) + O(t^{order+1 })`.

        Parameters
        ----------
        order : ``1, 2, 4, '4_opt'``
            The desired order of the Suzuki-Trotter decomposition.
            Order ``1`` approximation is simply :math:`e^A a^B`.
            Order ``2`` is the "leapfrog" `e^{A/2} e^B e^{A/2}`.
            Order ``4`` is the fourth-order from :cite:`suzuki1991` (also referenced in
            :cite:`schollwoeck2011`), and ``'4_opt'`` gives the optmized version of Equ. (30a) in
            :cite:`barthel2020`.

        Returns
        -------
        ST_decomposition : list of (int, int)
            Indices ``j, k`` of the time-steps ``d = suzuki_trotter_time_step(order)`` and
            the decomposition of `H`.
            They are chosen such that a subsequent application of ``exp(d[j] t H[k])`` to a given
            state ``|psi>`` yields ``(exp(N_steps t H[k]) + O(N_steps t^{order+1}))|psi>``.
        """
        even, odd = 0, 1
        if N_steps == 0:
            return []
        if order == 1:
            a = (0, odd)
            b = (0, even)
            return [a, b] * N_steps
        elif order == 2:
            a = (0, odd)  # dt/2
            a2 = (1, odd)  # dt
            b = (1, even)  # dt
            # U = [a b a]*N
            #   = a b [a2 b]*(N-1) a
            # return [a, b] + [a2, b] * (N_steps - 1) + [a]
            return [a, b, a] * N_steps
        elif order == 4:
            a = (0, odd)  # t1/2
            a2 = (1, odd)  # t1
            b = (1, even)  # t1
            c = (2, odd)  # (t1 + t3) / 2 == (1 - 3 * t1)/2
            d = (3, even)  # t3 = 1 - 4 * t1
            # From Schollwoeck 2011 (:arxiv:`1008.3477`):
            # U = U(t1) U(t2) U(t3) U(t2) U(t1)
            # with U(dt) = U(dt/2, odd) U(dt, even) U(dt/2, odd) and t1 == t2
            # Using above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            # steps = [a, b, a2, b, c, d, c, b, a2, b]
            # steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            # steps = steps + [a]
            # return steps
            return [a, b, a2, b, c, d, c, b, a2, b, a] * N_steps
        elif order == '4_opt':
            # symmetric: a1 b1 a2 b2 a3 b3 a2 b2 a2 b1 a1
            steps = [(0, odd), (1, even), (2, odd), (3, even), (4, odd),  (5, even),
                     (4, odd), (3, even), (2, odd), (1, even), (0, odd)]  # yapf: disable
            return steps * N_steps
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        """Calculate ``self.U_bond`` from ``self.bond_eig_{vals,vecs}``.

        This function calculates

        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.

        For first order (in `delta_t`), we need just one ``dt=delta_t``.
        Higher order requires smaller `dt` steps, as given by :meth:`suzuki_trotter_time_steps`.

        Parameters
        ----------
        order : int
            Trotter order calculated U_bond. See update for more information.
        delta_t : float
            Size of the time-step used in calculating U_bond
        type_evo : ``'imag' | 'real'``
            Determines whether we perform real or imaginary time-evolution.
        E_offset : None | list of float
            Possible offset added to `H_bond` for real-time evolution.
        """
        U_param = dict(order=order, delta_t=delta_t, type_evo=type_evo, E_offset=E_offset)
        if type_evo == 'real':
            U_param['tau'] = delta_t
        elif type_evo == 'imag':
            U_param['tau'] = -1.j * delta_t
        else:
            raise ValueError("Invalid value for `type_evo`: " + repr(type_evo))
        # if self._U_param == U_param:  # same keys and values as cached
        #     if self.verbose >= 10:
        #         print("Skip recalculation of U with same parameters as before: ", U_param)
        #     return  # nothing to do: U is cached
        self._U_param = U_param

        L = self.psi.L
        self._U = []
        # returns [prefactor of timestep for odd, prefactor of timestep for even]
        # for order 2 this is [.5, 1]
        # _U has shape (order, num_sites, U_bond shape)
        # so _U[0] -> odd U_bonds, _U[1] -> even U_bonds
        # _U[odd][i] gives bond between site (i-1, i), so there is none at 0 unless infinite bc
        for dt in self.suzuki_trotter_time_steps(order):
            U_bond = [
                self._calc_U_bond(i_bond, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
            ]
            self._U.append(U_bond)
        # done

    def update(self, N_steps, delta_t, tracking_current):
        """Evolve by ``N_steps * U_param['dt']``.

        Parameters
        ----------
        N_steps : int
            The number of steps for which the whole lattice should be updated.
        tracking_current : np.array
            The current we would like to track

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation during
            this sequence of update steps.
        """
        trunc_err = TruncationError()
        order = self._U_param['order']
        ti = time.time()
        times = [self.time]
        energies = [np.sum(self.model.bond_energies(self.psi))]
        currents = [self.currentop.H_MPO.expectation_value(self.psi)]
        phis = [0.]  # for tracking purposes
        i = 0  # for keeping track of when a timestep completes
        # returns [(0, odd boolean), (1, even_boolean), (0, odd_boolean)] * N
        # boolean is actually just an integer 0 - false, 1 - true
        for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
            i += 1
            trunc_err += self.update_step(U_idx_dt, odd)
            # """HERE WE WILL CALCULATE EXPECTATION VALUES"""
            # the if statement indicates that one time step of the order 2
            # method ONLY has completed
            if i % 3 == 0:
                self.time += delta_t
                t = time.time() - ti  # time simulation has been running
                complete = i / (N_steps * 3)  # proportion complete (2nd order)
                seconds = ((3 * t) / i) * (1 - complete) * N_steps  # time remaining
                days = int(seconds // (3600 * 24))
                seconds = seconds % (3600 * 24)
                hrs = int(seconds // 3600)
                seconds = seconds % 3600
                mins = int(seconds // 60)
                seconds = int(seconds % 60)
                status = "Simulation status: {:.2f}% -- ".format(complete * 100)
                status += "Estimated time remaining: {} days, {}".format(days, datetime.time(hrs, mins, seconds))
                print(status, end="\r")
                times.append(self.time)
                energies.append(np.sum(self.model.bond_energies(self.psi)))
                currents.append(self.currentop.H_MPO.expectation_value(self.psi))
                # now we must update the model which describes the hamiltonian
                # and the time evolution operator for the next step
                if tracking_current is not None:
                    phi = self.update_operators(tracking_current[int(i / 3)])
                    phis.append(phi)
                else:
                    self.update_operators(None)
                self.calc_U(order, delta_t, type_evo='real', E_offset=None)
        # print()
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err, np.array(times), np.array(energies), np.array(currents), np.array(phis)

    def update_operators(self, tcurrent):
        """
        Update the Hamiltonian and the Current operator so they correspond to the correct time
        """
        del self.model
        del self.currentop
        args = []
        if tcurrent is not None:
            args = [tcurrent, self]
        phi = self.phi_func(self.time, self.p, *args)
        model = FHHamiltonian(self.p, phi)
        currentop = FHCurrent(self.p, phi)
        self.model = model
        self.currentop = currentop
        return phi


    def update_step(self, U_idx_dt, odd):
        """Updates either even *or* odd bonds in unit cell.

        Depending on the choice of p, this function updates all even (``E``, odd=False,0)
        **or** odd (``O``) (odd=True,1) bonds::

        |     - B0 - B1 - B2 - B3 - B4 - B5 - B6 -
        |       |    |    |    |    |    |    |
        |       |    |----|    |----|    |----|
        |       |    |  E |    |  E |    |  E |
        |       |    |----|    |----|    |----|
        |       |----|    |----|    |----|    |
        |       |  O |    |  O |    |  O |    |
        |       |----|    |----|    |----|    |

        Note that finite boundary conditions are taken care of by having ``Us[0] = None``.

        Parameters
        ----------
        U_idx_dt : int
            Time step index in ``self._U``,
            evolve with ``Us[i] = self.U[U_idx_dt][i]`` at bond ``(i-1,i)``.
        odd : bool/int
            Indication of whether to update even (``odd=False,0``) or even (``odd=True,1``) sites

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation
            during this sequence of update steps.
        """
        Us = self._U[U_idx_dt]
        trunc_err = TruncationError()
        for i_bond in np.arange(int(odd) % 2, self.psi.L, 2):
            if Us[i_bond] is None:
                continue  # handles finite vs. infinite boundary conditions
            self._update_index = (U_idx_dt, i_bond)
            trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err

    def update_bond(self, i, U_bond):
        """Updates the B matrices on a given bond.

        Function that updates the B matrices, the bond matrix s between and the
        bond dimension chi for bond i. The correponding tensor networks look like this::

        |           --S--B1--B2--           --B1--B2--
        |                |   |                |   |
        |     theta:     U_bond        C:     U_bond
        |                |   |                |   |

        Parameters
        ----------
        i : int
            Bond index; we update the matrices at sites ``i-1, i``.
        U_bond : :class:`~tenpy.linalg.np_conserved.Array`
            The bond operator which we apply to the wave function.
            We expect labels ``'p0', 'p1', 'p0*', 'p1*'``.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced by the truncation
            during this update step.
        """
        i0, i1 = i - 1, i
        # Construct the theta matrix
        C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        C.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        # now theta is the same as if we had done
        #   theta = self.psi.get_theta(i0, n=2)
        #   theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # but also have C which is the same except the missing "S" on the left
        # so we don't have to apply inverses of S (see below)

        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    [self.psi.get_B(i0, None).qtotal, None],
                                                    inner_labels=['vR', 'vL'])

        # Split tensor and update matrices
        B_R = V.split_legs(1).ireplace_label('p1', 'p')

        # In general, we want to do the following:
        #     U = U.iscale_axis(S, 'vR')
        #     B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)**-1, 'vL')
        #     B_L = B_L.ireplace_label('p0', 'p')
        # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**-1 U S``
        #
        # However, the inverse of SL is problematic, as it might contain very small singular
        # values.  Instead, we use ``C == SL**-1 theta == SL**-1 U S V``,
        # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
        # here, C is the same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)
        B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(p1.vR)', '(p1*.vR*)'])
        B_L.ireplace_labels(['vL*', 'p0'], ['vR', 'p'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def _calc_U_bond(self, i_bond, dt, type_evo, E_offset):
        """Calculate exponential of a bond Hamitonian.

        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
        """
        h = self.model.H_bond[i_bond]
        if h is None:
            return None  # don't calculate exp(i H t), if `H` is None
        H2 = h.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
        if type_evo == 'imag':
            H2 = (-dt) * H2
        elif type_evo == 'real':
            if E_offset is not None:
                H2 = H2 - npc.diag(E_offset[i_bond], H2.legs[0])
            H2 = (-1.j * dt) * H2
        else:
            raise ValueError("Expect either 'real' or 'imag'inary time, got " + repr(type_evo))
        U = npc.expm(H2)
        assert (tuple(U.get_leg_labels()) == ('(p0.p1)', '(p0*.p1*)'))
        return U.split_legs()
