import cv2
import matplotlib.pyplot as plt

from random import randrange, uniform
from scipy.spatial.transform import Rotation as SciR

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

coord_root_global = (433, 470)  # x, y

img = cv2.imread("plant.bmp", cv2.IMREAD_GRAYSCALE)

# cv2.circle(img, coord_root_global, 20, 150, 3)
cv2.imshow('image', img)

n_points_back_up = 300
# img2 = img[(coord_root_global[1]-n_points_back_up):coord_root_global[1],
#    (coord_root_global[0]-50):(coord_root_global[0]+50)]
img2 = img[(coord_root_global[1]-n_points_back_up):coord_root_global[1], coord_root_global[0]]
img2_jnp = jnp.array(img2)

# print(img[coord_root_global[1]][coord_root_global[0]])

print(img2_jnp)


# def find_y_coord_min(y_column):
#     # y_column must be of a fixed length
#     for i in range(y_column)):
#         if (y_column[i][root_x_coord] == 0):
#             y_coord_min=i
#             break

# @ jit
# def sample_lines(coord_root_global, coord_root_global_lowest, mask_cropped, image_width, image_height):
#     pass

# img2 = img[0:400, 0:100]  # from top left, [row, column] [y, x]

# cv2.imshow('image', img2)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()

cv2.waitKey(2000)
cv2.destroyAllWindows()
