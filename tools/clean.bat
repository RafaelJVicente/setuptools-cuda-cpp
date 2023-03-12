pip uninstall my_cuda_package setuptools_cuda_cpp -y

rd /s /q .\build\
rd /s /q .\dist\
rd /s /q .\setuptools_cpp_cuda.egg-info\

rd /s /q .\examples\cuda_example\my_cuda_package.egg-info\
rd /s /q .\examples\cuda_example\my_cuda_package

pip install .
pip install .\examples\cuda_example\
