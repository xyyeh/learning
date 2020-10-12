import eigen as e
import rbdyn as rbd
import sva as s
import numpy as np
from rbdyn.parsers import *
from . import helper


class Arm:
    """
    A general 2 link arm description
    """

    def __init__(self, urdf):
        """
        Load arm from urdf
        @param urdf Path for urdf file
        """
        self.dyn = from_urdf_file(urdf)
        # gravity
        self.dyn.mbc.gravity = e.Vector3d(0, 0, 9.81)
        self.dyn.mbc.zero(self.dyn.mb)
        # jacobians
        self.J = e.MatrixXd(6, self.dyn.mb.nrDof())
        self.dJ = e.MatrixXd(6, self.dyn.mb.nrDof())

    def get_q(self):
        """
        Returns the joint angle q
        @param urdf Path for urdf file
        """
        q = np.zeros(self.dyn.mb.nrDof())
        for i in range(0, self.dyn.mb.nrDof()):
            q[i] = self.dyn.mbc.q[i+1][0]
        return q

    def set_q(self, q):
        """
        Sets the joint angles q
        @param q Joint angles
        """
        for i in range(0, self.dyn.mb.nrDof()):
            self.dyn.mbc.q[i+1][0] = q[i]
        return q

    def get_dq(self):
        """
        Returns the joint velocity dq
        @param urdf Path for urdf file
        """
        q = np.zeros(self.dyn.mb.nrDof())
        for i in range(0, self.dyn.mb.nrDof()):
            q[i] = self.dyn.mbc.q[i+1][0]
        return q

    def set_dq(self, q):
        """
        Sets the joint velocity q
        @param q Joint angles
        """
        for i in range(0, self.dyn.mb.nrDof()):
            self.dyn.mbc.q[i+1][0] = q[i]
        return q

    def forward_kine(self):
        """
        Computes forward kinematics
        """
        rbd.forwardKinematics(self.dyn.mb, self.dyn.mbc)
        rbd.forwardVelocity(self.dyn.mb, self.dyn.mbc)

    def sva_to_affine(self, s):
        """
        Converts a spatial transform matrix to a homogeneous transform matrix
        @param s Spatial transform
        @return Homogeneous transform matrix
        """
        m4d = e.Matrix4d.Identity()
        R = s.rotation().transpose()
        p = s.translation()

        for row in range(3):
            for col in range(3):
                m4d.coeff(row, col, R.coeff(row, col))
        for row in range(3):
            m4d.coeff(row, 3, p[row])

        return m4d

    def body_id_from_name(self, name, bodies):
        """
        Returns the body Id from the body name
        @param name The name of the body
        @param bodies The set of bodies provided by the multibody data structure
        @return Id of the body, -1 if not found
        """
        for bi, b in enumerate(bodies):
            if (b.name().decode("utf-8") == name):
                return bi
        return -1

    def get_body_trans(self, rot, pos, link_name):
        """
        Gets the body transformation matrix given a local SE3 offset on a particular link
        @param rot Rotation offset
        @param pos Position offset
        @param link_name Link name
        @return body SE3
        """
        ofsRot = e.Matrix3d(rot)
        ofsPos = e.Vector3d(pos)

        return self.sva_to_affine(s.PTransformd(ofsRot.transpose(), ofsPos) * self.dyn.mbc.bodyPosW[self.body_id_from_name(link_name, self.dyn.mb.bodies())])

    def get_jacobian_and_deriv(self, pos, link_name):
        """
        Gets jacobian and its time derivative given a local position offset on a particular link
        @param pos Position offset
        @param link_name Link name
        @return body SE3
        """
        jac = rbd.Jacobian(self.dyn.mb, link_name.encode('utf-8'), pos)
        # jac.fullJacobian(self.dyn.mb, jac.jacobianDot(
        #     self.dyn.mb, self.dyn.mbc), self.dJ)

        # nnz = np.count_nonzero(selection)
        # S = e.MatrixXd.Zero(nnz, 6)

        # i = 0
        # for j in range(len(selection)):
        #     if selection[j] == 1:
        #         S.coeff(i, j, 1)
        #         print(i, j)
        #         i = i+1

        # return S*J, S*dJ
        # return e.MatrixXd(6, 6), e.MatrixXd(6, 6)

    def get_mass_matrix(self):
        """
        Gets the mass matrix
        @return mass matrix
        """
        fd = rbd.ForwardDynamics(self.dyn.mb)
        fd.computeH(self.dyn.mb, self.dyn.mbc)
        return fd.H()

    def get_nonlinear_effects(self):
        """
        Gets the nonlinear effect torque vector
        @return nonlinear effects
        """
        fd = rbd.ForwardDynamics(self.dyn.mb)
        fd.computeC(self.dyn.mb, self.dyn.mbc)
        return fd.C()
