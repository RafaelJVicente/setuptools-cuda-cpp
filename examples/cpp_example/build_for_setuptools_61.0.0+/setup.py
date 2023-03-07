from pathlib import Path

from setuptools import setup

from setuptools_cpp_cuda import CppExtension, BuildExtension, fix_dll

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
    ext_modules=[cpp_ext],
    cmdclass={'build_ext': BuildExtension},
)
