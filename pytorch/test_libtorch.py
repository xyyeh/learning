import math
import torch
import pytorch_kinematics as pk

import os
from timeit import default_timer as timer

import time


# a = torch.tensor([ 0.47629452, -0.42736599, -0.21633042,  0.73736215])
# b = torch.tensor([ 0.69387621, 0.00774611, 0.02319487, 0.71967900])

# torch.set_printoptions(precision=8)

# # print(pk.quaternion_to_axis_angle(a))
# # print(pk.quaternion_to_axis_angle(b))

# print(pk.quaternion_to_axis_angle(torch.stack([a,b])))


# R = torch.tensor([[[ 0.7652395, -0.3224221, 0.5571826],
#                    [ 0.5571826, 0.7652395, -0.3224221],
#                    [ -0.3224221, 0.5571826, 0.7652395]]])

# print(R.shape)

# q = pk.matrix_to_quaternion(R)

chain = pk.build_serial_chain_from_urdf(open("./urdf/kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
chain = chain.to(torch.float32)

target_pos = torch.tensor([-6.6083e-01, -5.2764e-08, 3.7414e-01])
target_rot = torch.tensor([7.0711e-01, -4.3711e-08, -7.0711e-01, 6.1817e-08])


# m = torch.tensor([[1.0000, -0.0000, 0.0000, 0.0000],
#                   [0.0000,  1.0000, 0.0000, 0.0000],
#                   [-0.0000, 0.0000, 1.0000, 1.2610],
#                   [0.0000,  0.0000, 0.0000, 1.0000]])
# m = m.repeat(1,4,1,1)

torch.manual_seed(42)

num_tries = 400
num_iters = 40
pos_tolerance = 1e-3
rot_tolerance = 1e-2

exact = torch.tensor([ 0., -0.78539816339, 0., 1.57079632679, 0., 0.78539816339, 0. ]).repeat([num_tries, 1])
delta = 2. * (torch.rand_like(exact) - 0.5)
initial_guess = exact + delta

lim = torch.tensor(chain.get_joint_limits())
ik_cpu = pk.PseudoInverseIK(chain, 
                        pos_tolerance=pos_tolerance, 
                        rot_tolerance=rot_tolerance, 
                        max_iterations=num_iters, 
                        num_retries=num_tries,
                        joint_limits=lim.T,
                        early_stopping_any_converged=False,
                        early_stopping_no_improvement=None,
                        # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                        regularlization=1e-9,
                        retry_configs=initial_guess,
                        debug=False,
                        lr=1.0)

target_pose = pk.Transform3d(pos=target_pos, rot=target_rot)

t1 = timer()
sol = ik_cpu.solve(target_pose)
t2 = timer()

print("Execution time for IK, [parallel] in [cpu] = {}".format(t2-t1))

ik_cpu_sequential = pk.PseudoInverseIK(chain, 
                        pos_tolerance=pos_tolerance, 
                        rot_tolerance=rot_tolerance, 
                        max_iterations=num_iters, 
                        num_retries=1,
                        joint_limits=lim.T,
                        early_stopping_any_converged=False,
                        early_stopping_no_improvement=None,
                        # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                        regularlization=1e-9,
                        retry_configs=initial_guess,
                        debug=False,
                        lr=1.0)
t1 = timer()
for i in range(400):
    sol = ik_cpu.solve(target_pose)
t2 = timer()

print("Execution time for IK, [sequential] in [cpu] = {}".format(t2-t1))

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print("device = {}".format(d))
chain = chain.to(dtype=dtype, device=d)
target_pose = target_pose.to(dtype=dtype, device=d)
initial_guess = initial_guess.to(dtype=dtype, device=d)

ik_gpu = pk.PseudoInverseIK(chain, 
                        pos_tolerance=pos_tolerance, 
                        rot_tolerance=rot_tolerance, 
                        max_iterations=num_iters, 
                        num_retries=num_tries,
                        joint_limits=lim.T,
                        early_stopping_any_converged=False,
                        early_stopping_no_improvement=None,
                        # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                        regularlization=1e-9,
                        retry_configs=initial_guess,
                        debug=False,
                        lr=1.0)

t1 = timer()
sol = ik_gpu.solve(target_pose)
t2 = timer()

print("Execution time for IK, [parallel] in [gpu] = {}".format(t2-t1))
