import numpy as np


class Control(object):
    """
    Base class for controllers.
    """

    def __init__(self, kp=400, kv=np.sqrt(40), additions=[]):
        """
        Constructor
        @param addition list List of addition classes to overlay on the control signal
        @param kp Position gain
        @param kv Velocity gain
        """
        self.kp = kp
        self.kv = kv
        self.additions = additions
        self.target = 0

    def ReachedSE3(self, arm):
        """
        Checks that we reached an SE3 target using only 2-norm
        @param arm Handler to the arm state variables
        """
        return np.sum(abs(arm.x - self.target)) + np.sum(abs(arm.dq))
