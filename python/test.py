import control

import numpy as np
import scipy.linalg as sp_linalg


class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """

    def __init__(self, solve_continuous=False, **kwargs):
        super(Control, self).__init__(**kwargs)
        self.DOF = 2
        self.u = None
        self.solve_continuous = solve_continuous

    def calc_derivatives(self, x, u):
        eps = 1e-5
        x1 = np.tile(x, (self.arm.DOF*2, 1)).T + np.eye(self.arm.DOF*2) * eps
        x2 = np.tile(x, (self.arm.DOF*2, 1)).T - np.eye(self.arm.DOF*2) * eps
        uu = np.tile(u, (self.arm.DOF*2, 1))
        f1 = self.plant_dynamics(x1, uu)
        f2 = self.plant_dynamics(x2, uu)
        xdot_x = (f1 - f2) / 2 / eps
        return xdot_x, xdot_u

    def plant_dynamics(self, x, u):
        """
        Simulate plant dynamics locally
        """
        if x.ndim == 1:
            x = x[:, None]
            u = u[None, :]
