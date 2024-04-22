import math
import torch
import pytorch_kinematics as pk

import os
from timeit import default_timer as timer

import time

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

print("device = {}".format(d))

chain = pk.build_serial_chain_from_urdf(open("./urdf/kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
chain = chain.to(dtype=dtype, device=d)

N = 400
th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d, requires_grad=True)
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], dtype=dtype, device=d, requires_grad=True)

# ########################### FK ###########################

tg_batch = chain.forward_kinematics(th_batch)
tg_batch = chain.forward_kinematics(th_batch)
tg_batch = chain.forward_kinematics(th_batch)
tg_batch = chain.forward_kinematics(th_batch)
tg_batch = chain.forward_kinematics(th_batch)

# parallel fk
t1 = timer()
tg_batch = chain.forward_kinematics(th_batch)
t2 = timer()

# sequential fk
for i in range(N):
    tg = chain.forward_kinematics(th_batch[i])
t3 = timer()

J_batch = chain.jacobian(th_batch)
J_batch = chain.jacobian(th_batch)
J_batch = chain.jacobian(th_batch)
J_batch = chain.jacobian(th_batch)
J_batch = chain.jacobian(th_batch)

# parallel jac
t4 = timer()
J_batch = chain.jacobian(th_batch)
t5 = timer()

print("Execution time for FK, [parallel] = {}, [sequential] = {}".format(t2-t1, t3-t2))
print("Execution time for Jac, [parallel] = {}, [sequential] = {}".format(t5-t4, 0))

