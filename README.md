# Setuptools CUDA C++

[![PyPI - Version](https://img.shields.io/pypi/v/setuptools-cuda-cpp.svg)](https://pypi.org/project/setuptools-cuda-cpp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/setuptools-cuda-cpp.svg)](https://pypi.org/project/setuptools-cuda-cpp)
[![GitHub - Dependencies](https://img.shields.io/librariesio/release/pypi/setuptools-cuda-cpp?label=deps)](https://pypi.org/project/setuptools-cuda-cpp)
[![GitHub - Issues](https://img.shields.io/github/issues/RafaelJVicente/setuptools-cuda-cpp?color=yellow)](https://github.com/RafaelJVicente/setuptools-cuda-cpp/issues)
[![GitHub - Last commit](https://img.shields.io/github/last-commit/RafaelJVicente/setuptools-cuda-cpp?color=purple)](https://github.com/RafaelJVicente/setuptools-cuda-cpp)

[//]: # ([![GitHub - Build]&#40;https://img.shields.io/github/actions/workflow/status/RafaelJVicente/setuptools-cuda-cpp/unit-tests.yml&#41;]&#40;https://github.com/RafaelJVicente/setuptools-cuda-cpp&#41;)

The setuptools-cuda-cpp is a module that extends setuptools functionality for building hybrid C++ and CUDA extensions
for Python wrapper modules.

-----

**Table of Contents**

- [Summary](#summary)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Issues](#issues)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Summary

This project meant to be a soft solution to include mixed c++/CUDA extensions in your projects, no matter if you are
using old python version (3.6+) or old GPU drivers (sm/compute arch 3.0+).

## Features

- Python version >= 3.6 .
- SM(StreamingMultiprocessor)/Compute architecture >= 3.0 .
- Cython compatible but not mandatory.
- Any CUDA version (since you can configure nvcc flags).
- Preloaded flags for cpp and CUDA compilers.
- Mixed compilations (.cpp and .cu files can be included in a single extension).
- Advanced find_cuda features (automatically try to find the CUDAHOME directory).
- Include NVIDIA Management Library (NVML) capabilities info.

## Installation

```console
pip install setuptools-cuda-cpp
```

## Usage

Add the library to your project configuration files ("pyproject.toml" and/or "setup.py/.cfg").

### 1. Example for "legacy build" (old python versions with setuptools < 61.0.0):

[**setup.py**](./examples/cuda_example/setup.py)

```python
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
```

You can also use pyproject.toml with [Flit](https://flit.pypa.io) making
a [custom build-backend](https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks).

### 2. Example for "pyproject.toml build" (with setuptools >= 61.0.0):

[**pyproject.toml**](./examples/cuda_example/build_for_setuptools_61.0.0+/pyproject.toml)

```toml
[build-system]
requires = ["setuptools-cuda-cpp", "flit_core >=3.2,<4", "wheel", "cython"]
build-backend = "flit_core.buildapi"

[project]
name = "my-cuda-package"
dependencies = ["numpy"]
dynamic = ["version", "description"]
# ...
```

And configure the setup.py for the different extensions you want to use:

[**setup.py**](examples/cuda_example/build_for_setuptools_61.0.0+/setup.py)

```python
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
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},
)
```

## Issues

If you receive a EnvironmentError exception you should set CUDAHOME environment variable pointing to the CUDA
installation path. This would happen if the find_cuda() method is not capable of locate it.
As reference the directory should contain:

```text
CUDAHOME
├── bin
│   └── nvcc
├── include
│   └── cudart.h
├── lib
└── nvml
```

## License

`setuptools-cuda-cpp` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Acknowledgements
The package is based on [cpp_extension](https://pytorch.org/docs/stable/cpp_extension.html), but it also includes:
- Support for deprecated older architectures (from sm / compute 3.0).
- Improved find_cuda system.
- Pathlib library and Windows missing dll support.
