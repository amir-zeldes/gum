from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

cythonize("*.pyx")

setup(
    name = "hpsg_decoder",
    ext_modules = [Extension("hpsg_decoder",["hpsg_decoder.c"],include_dirs=[numpy.get_include()])]
)

setup(
    name="const_decoder",
    ext_modules = [Extension("const_decoder",["const_decoder.c"],include_dirs=[numpy.get_include()])]
)