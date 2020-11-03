from dmp_pkg.dmp import DMPs
import numpy as np


class DMPs_discrete(DMPs):
    """
    Discrete DMPs
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(DMPs_discrete, self).__init__(pattern="discrete", **kwargs)

        # generate centers for basis functions
        self.gen_centers()

        # set variance (sigma^2) of Gaussian basis functions
        self.h = (np.ones(self.n_bfs) * self.n_bfs ** 1.5) / \
            self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        """
        Generate centers of the Gaussian basis functions to be spaced evenly throughout run time

        Returns:
            an (n_bfs,) array containing all the centers of the basis function
        """
        # every interval is given by total run time divided by the number of basis functions
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        # find all the x for the linearly spaced times
        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_idx):
        """
        TODO: is this needed?
        Generates the diminishing term x(g-y0) of the forcing term

        Args:
            x (float): the phase variable, i.e. value of the canoncial system
            dmp_idx (int): the index in the dmp system

        Returns:
            x(g-y0)
        """
        return x * (self.goal[dmp_idx] - self.y0[dmp_idx])

    def gen_goal(self, y_des):
        """
        Generate goal for path imitation, for rhythmic DMPs, the goal is the average of the desired trajectory

        Args:
            y_des (array): the desired trajectory to follow, (n_dmps, n_points) array for n_dmp system

        Returns:
            goal of n_dmps as the last column of the y_des array
        """
        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """
        Generate the basis functions for a given canonical system rollout

        Args:
            x (float, array): the canonical system path rolled out or a system position

        Returns:
            a 2D NxM array with N being the length of rollout and M being the number of basis functions,
            each column is thus the evaluation of the i-th basis function psi_i for the rollout
            OR a 1xM array showing psi evaluated at the particular system position
        """
        if isinstance(x, np.ndarray):
            x = x[:, None]

        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        """
        Generates the set of weights so as to match the target forcing term trajectory

        Args:
            f_target (array): the desired forcing term trajectory, (n_dmps, n_points) array
        """

        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # calculate BF weights using LWR where w_i = \frac{s^T psi_i f_target}{s^T psi_i s}
        # for a particular dmp's i-th weight:
        # s = [x(t0)(g-y0) ... x(tN)(g-y0)]^T
        # psi_i = diag([psi_i(t0) .. psi_i(tN)])
        # f_target
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = self.goal[d] - self.y0[d]
            for b in range(self.n_bfs):
                # x_track = [x(t0) ... x(tN)] (N,1)
                # psi_track = [psi_0; psi_1; ... psi_i ... psi_(n_bfs-1)] (N,n_bfs) where
                # psi_i = [psi_i(t0) ... psi_i(tN)] (N,1) vector of basis function located center c_i evaluated at all of x_track
                # f_target = [f_d_dmp1; f_d_dmp2] (N,2)
                num = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                den = np.sum(x_track ** 2 * psi_track[:, b])
                # w is a (n_dmp, n_bfs) array
                self.w[d, b] = num / den
                if abs(k) > 1e-5:
                    # we just need to divide once since this scaling appears once on the num and square of it appears in denom
                    self.w[d, b] /= k

        # replace nan with meaningful value
        self.w = np.nan_to_num(self.w)


# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test normal run
    dmp = DMPs_discrete(dt=0.05, n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    y_track, dy_track, ddy_track = dmp.rollout()

    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track)) * dmp.goal, "r--", lw=2)
    plt.plot(y_track, lw=2)
    plt.title("DMP system - no forcing term")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["goal", "system state"], loc="lower right")
    plt.tight_layout()

    # test imitation of path run
    plt.figure(2, figsize=(6, 4))
    n_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    path1 = np.sin(np.arange(0, 1, 0.01) * 5)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.0):] = 0.5

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1, path2]))
        # change the scale of the movement
        dmp.goal[0] = 3
        dmp.goal[1] = 2

        y_track, dy_track, ddy_track = dmp.rollout()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(y_track[:, 0], lw=2)
        plt.subplot(212)
        plt.plot(y_track[:, 1], lw=2)

    plt.subplot(211)
    a = plt.plot(path1 / path1[-1] * dmp.goal[0], "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend([a[0]], ["desired path"], loc="lower right")
    plt.subplot(212)
    b = plt.plot(path2 / path2[-1] * dmp.goal[1], "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["%i BFs" % i for i in n_bfs], loc="lower right")

    plt.tight_layout()
    plt.show()
