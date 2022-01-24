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
img2 = img[(coord_root_global[1]-n_points_back_up):coord_root_global[1], coord_root_global[0]]
img2_jnp = jnp.array(img2)

cv2.waitKey(2000)
cv2.destroyAllWindows()
