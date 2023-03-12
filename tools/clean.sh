#!/bin/bash

cd "$(dirname $0)/.." || exit
pip uninstall my_cuda_package setuptools-cuda-cpp -y
rm -rf dist examples/cuda_example/build/ examples/cuda_example/my_cuda_package.egg-info/ examples/cuda_example/my_cuda_package
pip install .
pip install ./examples/cuda_example/
