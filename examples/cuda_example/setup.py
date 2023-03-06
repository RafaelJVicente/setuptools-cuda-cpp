from setuptools import setup

from setuptools_cpp_cuda import CudaExtension, BuildExtension, fix_dll

cuda_ext = CudaExtension(
    name='cudaext',
    include_dirs=['include'],
    sources=['cudaext.cu', 'cudaext_wrapper.cpp'],
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
