import sys
from pathlib import Path

IS_WINDOWS = sys.platform == 'win32'

which = 'where' if IS_WINDOWS else 'which'

SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()


def _is_cuda_file(path: str) -> bool:
    valid_ext = [
        '.cu',
        '.cuh',
    ]
    return Path(path).suffix in valid_ext
