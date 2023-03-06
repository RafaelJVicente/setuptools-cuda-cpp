from setuptools import setup

from setuptools_cpp_cuda import CppExtension, build_cuda_ext, fix_dll

cpp_ext = CppExtension(
    name='cudaext',
    include_dirs=['include'],
    sources=['cudaext.cu', 'cudaext_wrapper.cpp'],
    libraries=fix_dll(['cudart']),
    extra_compile_args={
        'cxx': ['-g'],
    },
)
setup(
    name='cudaext',
    ext_modules=[cpp_ext],
    cmdclass={'build_ext': build_cuda_ext},
)
