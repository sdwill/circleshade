# from circleshade import sample_disk, disk_artist, find_edges
# import numpy as np
# import matplotlib.pyplot as plt

# """
# Create a circle with a randomly-generated size and location, and show its discrete area-weghted
# representation and edge set.
# """
# N = 64
# dx = 1. / N

# # Randomly choose a radius and center location location, guaranteeing that the circle fits inside
# # the array
# radius = 0.25 * np.random.rand()
# xc = 0.5 * np.random.rand() - 0.25
# yc = 0.5 * np.random.rand() - 0.25

# # Build coordinate axes
# x = np.arange(-N / 2, N / 2) * dx
# y = np.arange(-N / 2, N / 2) * dx

# # Get disk and edge pixels
# disk = sample_disk(radius, (xc, yc), x)
# edges, interior, _ = find_edges(radius, (xc, yc), x)

# # Get the width of the array in physical units, treating each pixel as a square with finite width.
# # The physical width of the array extends from the left edge of the leftmost pixel in the array to
# # the right edge of the rightmost pixel.
# pixel_extent = (x.min() - 0.5 * dx, x.max() + 0.5 * dx,
#                 y.min() - 0.5 * dx, y.max() + 0.5 * dx)

# # Show disk
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
# im = axs[0].imshow(disk, 'gray', extent=pixel_extent)
# axs[0].add_artist(disk_artist(radius, (xc, yc)))
# axs[0].set_title('Shaded disk', y=1.01)

# # Show edges
# axs[1].imshow(edges, 'gray', extent=pixel_extent)
# axs[1].add_artist(disk_artist(radius, (xc, yc)))
# axs[1].set_title('Edges', y=1.01)

# print('Area of sampled disk:', np.sum(disk) * (dx ** 2))
# print('Area of real disk:   ', np.pi * (radius ** 2))
# print('Residual area:', np.pi * (radius ** 2) - np.sum(disk) * (dx ** 2))

# plt.show()
from circleshade import sample_disk
import numpy as np
import matplotlib.pyplot as plt

N = 64  # Number of pixels in array
dx = 1 / N  # Array pixel width
R = 28 * dx  # Radius
center = (1.3 * dx, 1.5 * dx)  # Center location
axis = np.arange(-0.5, 0.5, dx)

disk = sample_disk(R, center, axis)

plt.figure()
plt.imshow(disk)
plt.show()
