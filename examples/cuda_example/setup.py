from pathlib import Path

from setuptools import setup

from setuptools_cuda_cpp import CUDAExtension, BuildExtension, fix_dll

cuda_ext_path = Path('src/my_cuda_package/cuda_ext')
cuda_ext = CUDAExtension(
    name='my_cuda_package.cuda_ext',
    include_dirs=[cuda_ext_path / 'include'],
    sources=[
        cuda_ext_path / 'cuda_ext.cu',
        cuda_ext_path / 'cuda_ext_wrapper.cpp',
    ],
    libraries=fix_dll(['cudart']),  # Use fix_dll() only for Windows compatibility (check documentation for more info).
    extra_compile_args={
        'cxx': ['-g'],  # cpp compiler flags
        'nvcc': ['-O2'],  # nvcc flags
    },
)

setup(
    name='my-cuda-package',
    version='0.0.1',
    install_requires=['numpy', ],
    extras_require={'cython': ['cython'], },
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},
)
