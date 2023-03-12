import glob
import os
import subprocess
from pathlib import Path

from .utils import IS_WINDOWS, SUBPROCESS_DECODE_ARGS, which


def find_cuda_home() -> str:
    return str(find_cuda_home_path)


def find_cuda_home_path() -> Path:
    cuda_home = _find_cuda_home_path()
    if not cuda_home.exists() or not cuda_home.is_dir():
        raise EnvironmentError(
            f'CUDA_HOME environment inferred path {cuda_home.resolve()} not exist.'
            f' Please set CUDA_HOME environment variable to your CUDA install root ("installation_path/cuda")'
        )
    return cuda_home


def _find_cuda_home_path() -> Path:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is not None:
        return Path(cuda_home)

    try:
        with Path(os.devnull).open('w') as devnull:
            nvcc_path = subprocess.check_output([which, 'nvcc'], stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip(
                '\r\n')
            return Path(nvcc_path).parent.parent
    except Exception:
        pass

    if IS_WINDOWS:
        cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
        if len(cuda_homes) > 0:
            return Path(cuda_homes[0])

    cuda_home = Path('/usr/local/cuda')
    if cuda_home.exists():
        return cuda_home

    cuda_homes = glob.glob('/opt/nvidia/hpc_sdk/*/*/cuda')
    if len(cuda_homes) > 0:
        return Path(cuda_homes[0])

    nvcompilers = os.environ.get('NVCOMPILERS', '/opt/nvidia/hpc_sdk')
    nvarch = os.environ.get('NVARCH')
    cuda_homes = glob.glob(f'{nvcompilers}/{nvarch}/*/cuda')
    if len(cuda_homes) > 0:
        return Path(cuda_homes[0])

    raise EnvironmentError(
        f' Please set CUDA_HOME environment variable to your CUDA install root ("installation_path/cuda")'
    )
