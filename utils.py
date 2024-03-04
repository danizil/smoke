import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def sample_points_on_disc(radius, num_points):
    # Generate random radii (sqrt is used for uniform sampling)
    r = radius * np.sqrt(np.random.rand(num_points))
    # Generate random angles
    theta = 2 * np.pi * np.random.rand(num_points)
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.concatenate((x[:, None], y[:, None]), axis=1)



