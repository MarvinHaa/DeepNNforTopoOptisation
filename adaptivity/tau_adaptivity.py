import numpy as np


class tauadaptor():


    def __init__(self, tau_0, TOL=1e-2, TOL_t=1e2, k_P=0.075, k_I=0.175, k_D=0.01, tau_max=1e-3, tau_min=1e-8, tau_t_max=1e0, tau_t_min=1e-5):
        """
        Drive pseudo-time step tau by an evolutionary proportional integral derivative controller

        :param tau_0: strart value for tau in Lineat Elasticity Problem (LEP)
        :param TOL:
        :param k_P: given in paper WIAS
        :param k_I: given in paper WIAS
        :param k_D: given in paper WIAS
        :param tau_max: given in paper WIAS
        :param tau_min: given in paper WIAS
        """

        self.TOL = TOL
        self.k_P = k_P
        self.k_I = k_I
        self.k_D = k_D
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.tau_t_max = tau_t_max
        self.tau_t_min = tau_t_min
        self.TOL_t = TOL_t

        self.dim = len(tau_0)

        if self.dim == 1:
            self.taus = [tau_0]
        if self.dim == 2:
            self.taus = [tau_0]
        self.e_n_list = []

    def nextTau(self, e_n):

        if self.dim == 1:
            if len(self.e_n_list) > 2:
                tau_next = (self.e_n_list[-1][0] / e_n) ** self.k_P \
                           * (self.TOL / e_n) ** self.k_I * (self.e_n_list[-1] ** 2 / (e_n * self.e_n_list[-2])) ** self.k_D \
                            * self.taus[-1][0]

                tau_next = np.maximum(self.tau_min, tau_next[0])
                tau_next = np.maximum(self.taus[-1][0] / 2, tau_next)
                tau_next = np.minimum(self.tau_max, tau_next)
                tau_next = np.minimum(2 * self.taus[-1][0], tau_next)
            else:
                tau_next = self.taus[-1][0]

            self.taus.append(np.array([tau_next]))
            self.e_n_list.append(e_n)

            return np.array([tau_next])

        elif self.dim == 2:

            if len(self.e_n_list) > 2:
                tau_next_phi = (self.e_n_list[-1][0] / e_n[0]) ** self.k_P \
                           * (self.TOL / e_n[0]) ** self.k_I * (self.e_n_list[-1][0] ** 2 / (e_n[0] * self.e_n_list[-2][0])) ** self.k_D \
                           * self.taus[-1][0]

                tau_next_phi = np.maximum(self.tau_min, tau_next_phi)
                tau_next_phi = np.maximum(self.taus[-1][0] / 2, tau_next_phi)
                tau_next_phi = np.minimum(self.tau_max, tau_next_phi)
                tau_next_phi = np.minimum(2 * self.taus[-1][0], tau_next_phi)

                if e_n[1] != 0:

                    tau_next_t = (self.e_n_list[-1][1] / e_n[1]) ** self.k_P \
                                   * (self.TOL_t / e_n[1]) ** self.k_I * (self.e_n_list[-1][1] ** 2 / (e_n[1] * self.e_n_list[-2][1])) ** self.k_D \
                                   * self.taus[-1][1]

                    tau_next_t = np.maximum(self.tau_t_min, tau_next_t)
                    tau_next_t = np.maximum(self.taus[-1][1] / 2, tau_next_t)
                    tau_next_t = np.minimum(self.tau_t_max, tau_next_t)
                    tau_next_t = np.minimum(2 * self.taus[-1][1], tau_next_t)
                else:
                    tau_next_t = self.taus[-1][1]
            else:
                tau_next_phi = self.taus[-1][0]
                tau_next_t = self.taus[-1][1]




            self.taus.append(np.array([tau_next_phi, tau_next_t]))
            self.e_n_list.append(e_n)

            return self.taus[-1]
