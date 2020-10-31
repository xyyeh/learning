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
                timesteps = int(self.timesteps / kwargs["tau"])
            else:
                timesteps = self.timesteps

        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros_like(y_track)
        ddy_track = np.zeros_like(y_track)

        for t in range(timesteps):
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def gen_front_term(self, x, dmp_num):
        raise NotImplementedError()

    def gen_goal(self, y_des):
        raise NotImplementedError()

    def gen_psi(self):
        raise NotImplementedError()

    def gen_weights(self, f_target):
        raise NotImplementedError()

    def step(self, tau=1.0, error=0.0, ext_force=None):
        """
        Step the DMP system

        Args:
            tau (float, optional): [description]. Defaults to 1.0.
            error (float, optional): [description]. Defaults to 0.0.
            ext_force ([type], optional): [description]. Defaults to None.
        """

        error_coupling = 1/(1+error)

        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):
            # forcing term
            f = self.gen_front_term(
                x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)
            # acceleration
            self.ddy[d] = (1.0/tau)*self.ay[d]*(self.by[d] *
                                                (self.goal[d]-self.y[d])-self.dy[d])
            if ext_force is not None:
                self.ddy[d] += ext_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy
