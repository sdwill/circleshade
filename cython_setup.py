import distutils
from Cython.Build import cythonize
import numpy

distutils.core.setup(
    ext_modules=cythonize("circleshade/internal/disk.pyx"),
    include_dirs=[numpy.get_include()]
)
