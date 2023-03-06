# Setuptools C++ CUDA

[![PyPI - Version](https://img.shields.io/pypi/v/setuptools-cpp-cuda.svg)](https://pypi.org/project/setuptools-cpp-cuda)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/setuptools-cpp-cuda.svg)](https://pypi.org/project/setuptools-cpp-cuda)

The setuptools-cpp-cuda is a module that extends setuptools functionality for building hybrid C++ and CUDA extensions
for Python wrapper modules.

-----

**Table of Contents**

- [Summary](#summary)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Issues](#issues)
- [License](#license)

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
pip install setuptools-cpp-cuda
```

## Usage

Add the library to your project configuration files ("pyproject.toml", "setup.py/.cfg" or "requirements.txt"):

```toml
# [pyproject.toml]
[build-system]
requires = ["setuptools-cpp-cuda"]
# ...
```

```shell
# [requirements.txt]
setuptools-cpp-cuda
```

And configure the setup.py for the different extensions you want to use:

```python
# [setup.py]
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

`setuptools-cpp-cuda` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
