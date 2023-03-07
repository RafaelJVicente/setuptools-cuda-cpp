import os
from pathlib import Path
from typing import List, Union

import setuptools

from .find_cuda import find_cuda_home_path
from .utils import IS_WINDOWS

CUDA_HOME = find_cuda_home_path()
cudnn_path = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
CUDNN_HOME = Path(cudnn_path) if cudnn_path is not None else None


PathLike = Union[str, Path]

def CppExtension(name: str, sources: List[PathLike], *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> setup(
                name='extension',
                ext_modules=[
                    CppExtension(
                        name='extension',
                        sources=['extension.cpp'],
                        extra_compile_args=['-g']),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''
    kwargs['language'] = 'c++'
    return _prepare_extension(name, sources, *args, **kwargs)


def CudaExtension(name: str, sources: List[PathLike], *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> setup(
        ...     name='cuda_extension',
        ...     ext_modules=[
        ...         CUDAExtension(
        ...                 name='cuda_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cu'],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'nvcc': ['-O2']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    Relocatable device code linking:
    If you want to reference device symbols across compilation units (across object files),
    the object files need to be built with `relocatable device code` (-rdc=true or -dc).
    An exception to this rule is "dynamic parallelism" (nested kernel launches)  which is not used a lot anymore.
    `Relocatable device code` is less optimized so it needs to be used only on object files that need it.
    Using `-dlto` (Device Link Time Optimization) at the device code compilation step and `dlink` step
    help reduce the protentional perf degradation of `-rdc`.
    Note that it needs to be used at both steps to be useful.
    If you have `rdc` objects you need to have an extra `-dlink` (device linking) step before the CPU symbol linking step.
    There is also a case where `-dlink` is used without `-rdc`:
    when an extension is linked against a static lib containing rdc-compiled objects
    like the [NVSHMEM library](https://developer.nvidia.com/nvshmem).
    Note: Ninja is required to build a CUDA Extension with RDC linking.
    Example:
        >>> CUDAExtension(
        ...        name='cuda_extension',
        ...        sources=['extension.cpp', 'extension_kernel.cu'],
        ...        dlink=True,
        ...        dlink_libraries=["dlink_lib"],
        ...        extra_compile_args={'cxx': ['-g'],
        ...                            'nvcc': ['-O2', '-rdc=true']})
    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += cuda_library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    if not any(map(lambda s: s.startswith('cudart'), libraries)):
        libraries.append('cudart')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += cuda_include_paths()
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = list(kwargs.get('dlink_libraries', []))
    if (kwargs.get('dlink', False)) or len(dlink_libraries) > 0:
        extra_compile_args = dict(kwargs.get('extra_compile_args', {}))

        extra_compile_args_dlink = list(extra_compile_args.get('nvcc_dlink', []))
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return _prepare_extension(name, list(sources), *args, **kwargs)


def _prepare_extension(name: str, sources: List[PathLike], *args, **kwargs):
    name = str(name)
    sources = list(map(str, sources))
    kwargs['library_dirs'] = list(map(str, kwargs.get('library_dirs', [])))
    kwargs['libraries'] = list(map(str, kwargs.get('libraries', [])))
    kwargs['include_dirs'] = list(map(str, kwargs.get('include_dirs', [])))
    kwargs['extra_compile_args'] = list(map(str, kwargs.get('extra_compile_args', [])))

    return setuptools.Extension(name, sources, *args, **kwargs)


def cuda_include_paths() -> List[Path]:
    paths = []
    cuda_home_include = CUDA_HOME / 'include'
    # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
    # but gcc doesn't like having /usr/include passed explicitly
    if cuda_home_include != Path('/usr/include'):
        paths.append(cuda_home_include)
    if CUDNN_HOME is not None:
        paths.append(CUDNN_HOME / 'include')
    return paths


def cuda_library_paths() -> List[Path]:
    paths = []
    if IS_WINDOWS:
        lib_dir = Path('lib') / 'x64'
    else:
        lib_dir = 'lib64'
        if not (CUDA_HOME / lib_dir).exists() and (CUDA_HOME / 'lib').exists():
            # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
            # Note that it's also possible both don't exist (see
            # _find_cuda_home) - in that case we stay with 'lib64'.
            lib_dir = 'lib'
    paths.append(CUDA_HOME / lib_dir)

    if CUDNN_HOME is not None:
        paths.append(CUDNN_HOME / lib_dir)

    return paths
