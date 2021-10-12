

class Parameters:
    """
    Scales parameters to atomic units in terms of t_0.
    input units: eV (t, U)
    """

    def __init__(self, nsites, u, t, bc):
        self.nsites = nsites
        self.bc = bc
        self.nup = nsites // 2 + nsites % 2
        self.ndown = nsites // 2

        self.u = u / t
        self.t = 1.
