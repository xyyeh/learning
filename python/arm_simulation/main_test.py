# import hello

# hello.my_function()
# print(hello.name)

# nicholas = hello.Student("xy", "cs")
# nicholas.get_student_details()

import eigen as e
# import arm_pkg.arm
import numpy as np

from arm_pkg import arm

a = arm.Arm("urdf/four_dof.urdf")
b = np.random.rand(4)

print(b)
a.set_q(b)
print(a.get_q())

rot = np.identity(3)
pos = np.array([1, 0, 0])

print(a.get_body_trans(rot, pos, "link_4"))

a.forward_kine()
print(a.get_mass_matrix())
print(a.get_nonlinear_effects())
