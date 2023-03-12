import os
from pathlib import Path
from typing import List, Iterable

import setuptools

from .find_cuda import find_cuda_home_path
from .utils import IS_WINDOWS, PathLike

CUDA_HOME = find_cuda_home_path()
cudnn_path = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
CUDNN_HOME = Path(cudnn_path) if cudnn_path is not None else None


def CppExtension(name: str, sources: Iterable[PathLike], *args, **kwargs):
    r"""
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
    """
    # if 'language' not in kwargs:
    kwargs['language'] = 'c++'
    return _prepare_extension(name, sources, *args, **kwargs)


def CUDAExtension(name: str, sources: Iterable[PathLike], *args, **kwargs):
    r"""
    Example:
        >>> setup(
        ...     name='cuda_extension',
        ...     ext_modules=[
        ...         CUDAExtension(
        ...                 name='cuda_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cu'],
        ...                 dlink=True,
        ...                 dlink_libraries=["dlink_lib"],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'nvcc': ['-O2', '-rdc=true']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """
    library_dirs = list(kwargs.get('library_dirs', []))
    library_dirs += cuda_library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = list(kwargs.get('libraries', []))
    if not any(map(lambda s: s.startswith('cudart'), libraries)):
        libraries.append('cudart')
    kwargs['libraries'] = libraries

    include_dirs = list(kwargs.get('include_dirs', []))
    include_dirs += cuda_include_paths()
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = list(kwargs.get('dlink_libraries', []))
    if (kwargs.get('dlink', False)) or len(dlink_libraries) > 0:
        extra_compile_args = kwargs.get('extra_compile_args', {})

        extra_compile_args_dlink = list(extra_compile_args.get('nvcc_dlink', []))
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return _prepare_extension(name, sources, *args, **kwargs)


def _prepare_extension(name: str, sources: Iterable[PathLike], *args, **kwargs):
    name = str(name)
    sources = list(map(str, sources))
    kwargs['library_dirs'] = list(map(str, kwargs.get('library_dirs', [])))
    kwargs['libraries'] = list(map(str, kwargs.get('libraries', [])))
    kwargs['include_dirs'] = list(map(str, kwargs.get('include_dirs', [])))

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
        lib_dir = 'lib/x64'
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
