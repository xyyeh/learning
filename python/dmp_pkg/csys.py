import numpy as np


class CanonicalSystem(object):
    """
    Canonical system
    """

    def __init__(self, dt, ax=5.0, pattern="discrete"):
        """
        Constructor

        Args:
            dt (float): Time step
            ax (float, optional): Gain term in dynamical system. Defaults to 1.0.
            pattern (str, optional): Pattern of dmp, either "discrete" or "rhythmic". Defaults to "discrete".
        """
        self.ax = ax
        self.pattern = pattern
        if pattern == "discrete":
            self.run_time = 1.0
            self.step = self.step_discrete
        else:
            raise Exception("Invalid pattern")

        self.dt = dt
        # default time steps
        self.default_time_steps = int(self.run_time / self.dt)

        self.reset()

    def reset(self):
        """
        Reset system state, i.e. set x to 1
        """
        self.x = 1.0

    def step_discrete(self, tau=1.0, error_coupling=1.0):
        """
        Generate a single step of x for discrete movements. Decaying from 1 to 0 according to tau*dx = -ax*x (see 2.2)

        Args:
            tau (float, optional): Time constant, increase tau to make system execute slower. Defaults to 1.0.
            error_coupling (float, optional): Slow down if |error| is > 0. Defaults to 1.0.
        """
        self.x += (-self.ax * self.x * error_coupling / tau) * self.dt
        return self.x

    def rollout(self, **kwargs):
        """
        Generate x for open loop movements
        """
        if "tau" in kwargs:
            # if time constant is increase, use more time steps
            time_steps = int(self.default_time_steps * kwargs["tau"])
        else:
            time_steps = self.default_time_steps
        self.x_track = np.zeros(time_steps)

        self.reset()
        for t in range(time_steps):
            self.x_track[t] = self.x
            self.step(**kwargs)

        return self.x_track
