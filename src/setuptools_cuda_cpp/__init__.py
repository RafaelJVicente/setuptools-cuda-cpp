"""
Module that extends setuptools functionality for building hybrid C++ and CUDA extension for Python wrapper modules.
"""
from .build_ext import BuildExtension, fix_dll
from .extension import CppExtension, CUDAExtension, CUDA_HOME, CUDNN_HOME
from .find_cuda import find_cuda_home, find_cuda_home_path

__version__ = '0.1.7'
__all__ = [
    'BuildExtension', 'CppExtension', 'CUDAExtension',
    'find_cuda_home', 'find_cuda_home_path',
    'fix_dll', 'nvml'
]
