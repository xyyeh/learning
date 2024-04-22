import numpy as np

# a = np.random.rand(3,4)
# b = np.random.rand(4,1)


# for i in range(3):
#     for j in range(4):
#         c[i][j] = a[i][j] + b[j] 

# print(c)

yp = 0.9
y = 1.
print(-y*np.log(yp)-(1-y)*np.log(1-yp))

a = np.random.randn(1,3)
b = np.random.randn(3,3)
c = a*b
print(c)


print(a)