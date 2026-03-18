from setuptools import setup, Extension
import numpy

ext = Extension(
    'symnmfmodule',
    sources=['symnmfmodule.c', 'symnmf.c'],
    include_dirs=[numpy.get_include()],
)

setup(name='symnmfmodule', ext_modules=[ext])