from pathlib import Path

from setuptools import setup

from setuptools_cuda_cpp import CppExtension, BuildExtension, fix_dll

cpp_ext_path = Path('src/my_cpp_package/cppest')
cpp_ext = CppExtension(
    name='my_cpp_package.cppest',
    include_dirs=[cpp_ext_path / 'include'],
    sources=[
        cpp_ext_path / 'cppest.cpp',
        cpp_ext_path / 'cppest_wrapper.cpp',
    ],
    libraries=fix_dll([]),  # Use fix_dll() only for Windows compatibility (check documentation for more info).
    extra_compile_args={
        'cxx': ['-g'],  # cpp compiler flags
        'nvcc': ['-O2'],  # nvcc flags
    },
)

setup(
    name='my-cpp-package',
    version='0.0.1',
    install_requires=['numpy', ],
    extras_require={'cython': ['cython'], },
    ext_modules=[cpp_ext],
    cmdclass={'build_ext': BuildExtension},
)
