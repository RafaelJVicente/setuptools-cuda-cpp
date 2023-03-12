from pathlib import Path

from setuptools import setup

from setuptools_cpp_cuda import CUDAExtension, BuildExtension, fix_dll

cuda_ext_path = Path('src/my_cuda_package/cudaest')
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
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},
)
