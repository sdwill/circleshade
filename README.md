# circleshade
Exact area-weighted antialiasing of circular disks.  Implementation of the algorithm described in
Will and Fienup, "*An algorithm for exact area-weighted antialiasing of discrete circular apertures,*"
2019 (submitted).

# Requirements
NumPy, SciPy, matplotlib, Cython

# Installation
To install the latest version from Github:
```
git clone https://github.com/sdwill/circleshade
cd circleshade
python setup.py develop
python cython_setup.py build_ext --inplace
```

# Example usage
After installing and building:

```python
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
```

# Known issues
- If the requested disk exceeds the boundaries of the computational array, the algorithm will fail
  due to an out of bounds error
