import numpy as np


a = np.array([[1, 2, 3, 4, 5]]).T
print(a.shape)

aa = np.tile(a, (2, 1))

print(aa.shape)

# print(a[:].shape)
# print(a[:, None].shape)

# print(np.zeros(a.shape))
