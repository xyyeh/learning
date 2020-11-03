import numpy as np
from dmp_pkg.csys import CanonicalSystem

TOLERANCE = 1e-5


class DMPs(object):
    """
    Dmp as described in Schaal's paper, "Dynamical movement primitives: learning attractor models for motor behaviors"
    """

    def __init__(self, n_dmps, n_bfs, dt=0.01, y0=0, goal=1, w=None, ay=None, by=None, **kwargs):
        """
        Constructor for generic DMPs containing several transformation system

        Args:
            n_dmps (int): number of DMPs
            n_bfs (int): number of basis functions per DMP
            dt (float, optional): time step for simulation. Defaults to 0.01.
            y0 (int, optional): initial state of system. Defaults to 0.
            goal (int, optional): goal state of system. Defaults to 1.
            w (list, optional): weights for basis functions. Defaults to None.
            ay (float, optional): alpha gain of attractor. Defaults to None.
            by (float, optional): beta gain of attractor. Defaults to None.
        """
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt

        self.y0 = np.ones(self.n_dmps) * y0
        self.goal = np.ones(self.n_dmps) * goal

        if w is None:
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

        self.ay = np.ones(self.n_dmps) * 25.0 if ay is None else ay
        self.by = self.ay / 4.0 if by is None else by

        # setup canonical system
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time/self.dt)

        # setup DMPs
        self.reset()

    def reset(self):
        """
        Resets the states of all DMPs
        """
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()

    def check_offset(self):
        """
        Checks to see initial and goal are the same. If so, add a small offset so as to prevent the forcing term from being 0
        """
        # TODO: need to find out why we cannot handle unforced systems
        for i in range(self.n_dmps):
            if abs(self.y0[i]-self.goal[i]) < TOLERANCE:
                self.goal[i] += TOLERANCE

    def rollout(self, timesteps=None, **kwargs):
        """
        Generate a system trial, no feedback is incorporated

        Args:
            timesteps ([type], optional): [description]. Defaults to None.
        """
        self.reset()

        if timesteps is None:
            if "tau" in kwargs:
              # if time constant is increase, use more time steps
                timesteps = int(self.timesteps * kwargs["tau"])
            else:
                timesteps = self.timesteps

        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def gen_front_term(self, x, dmp_num):
        raise NotImplementedError()

    def gen_goal(self, y_des):
        raise NotImplementedError()

    def gen_psi(self, x):
        raise NotImplementedError()

    def gen_weights(self, f_target):
        raise NotImplementedError()

    def step(self, tau=1.0, L2_error=0.0, ext_force=None):
        """
        Step the DMP system

        Args:
            tau (float, optional): [description]. Defaults to 1.0.
            error (float, optional): [description]. Defaults to 0.0.
            ext_force ([type], optional): [description]. Defaults to None.

        Returns:
            next step of y, dy and ddy of all dmps
        """

        # when error is large, we slow down the integration
        error_coupling = 1/(1+L2_error)

        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # evaluate basis functions at this x
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):
            # goal directed forcing term
            f = self.gen_front_term(
                x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)

            # total forces
            f_total = f
            if ext_force is not None:
                f_total += ext_force[d]
            # acceleration, tau > 1 to slow down
            self.ddy[d] = (
                self.ay[d]*(self.by[d]*(self.goal[d]-self.y[d])-self.dy[d])+f_total)

            # slow down if there is error
            self.dy[d] += self.ddy[d] * (error_coupling / tau) * self.dt
            self.y[d] += self.dy[d] * (error_coupling / tau) * self.dt

        return self.y, self.dy, self.ddy

    def imitate_path(self, y_des, tau=1.0, plot=False):
        """
        Takes in a desired trajectory and generate the system parameters

        Args:
            y_des (array): Desired trajectory (only position coordinates is required), in an (n_dmps, n_points) array
            plot (bool, optional): Plots alll the DMPs. Defaults to False.

        Return:
            interpolated y_des as an (n_dmps, timesteps)
        """
        # initial state and goal of all dmps
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)
        n_y_des_pts = y_des.shape[1]

        # check offset to see if y0 is close to g, which may zero out forcing term

        # interpolate desired trajectory and replace y_des
        import scipy.interpolate
        # timesteps = runtime/dt of cs
        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, n_y_des_pts)
        for d in range(self.n_dmps):
            interp_func = scipy.interpolate.interp1d(x, y_des[d])
            for i in range(self.timesteps):
                path[d, i] = interp_func(i * self.dt)
        y_demo = path

        # generate acceleration and velocity
        dy_demo = np.gradient(y_demo, axis=1) / self.dt
        ddy_demo = np.gradient(dy_demo, axis=1) / self.dt

        # # set initial acceleration and velocity to be 0
        # for d in range(self.n_dmps):
        #     dy_demo[d, 0] = 0
        #     ddy_demo[d, 0] = 0

        # desired forcing term, an array of (n_points, n_dmps)
        f_target = np.zeros((self.timesteps, self.n_dmps))
        for d in range(self.n_dmps):
            f_target[:, d] = tau * tau * ddy_demo[d] - self.ay[d] * \
                (self.by[d] * (self.goal[d] - y_demo[d]) - tau * dy_demo[d])

        # find weights
        self.gen_weights(f_target)

        # plot
        if plot is True:
            # plot the basis function activations
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(211)
            psi_track = self.gen_psi(self.cs.rollout())
            plt.plot(psi_track)
            plt.title("basis functions")

            # plot the desired forcing function vs approx
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(f_target[:, ii], "--", label="f_target %i" % ii)
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                print("w shape: ", self.w.shape)
                plt.plot(
                    np.sum(psi_track * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.reset()

        return y_des
