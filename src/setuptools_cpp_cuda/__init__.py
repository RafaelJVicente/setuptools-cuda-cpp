"""
The setuptools-cpp-cuda is a module that extends setuptools functionality for building hybrid C++ and CUDA extensions
for Python wrapper modules.
"""
from .build_ext import BuildExtension, fix_dll
from .extension import CppExtension, CudaExtension, CUDA_HOME, CUDNN_HOME
from .find_cuda import find_cuda_home, find_cuda_home_path

__version__ = '0.1.1'
__all__ = [
    'BuildExtension', 'CppExtension', 'CudaExtension',
    'find_cuda_home', 'find_cuda_home_path',
    'fix_dll', 'nvml'
]
