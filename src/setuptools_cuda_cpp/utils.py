import sys
from pathlib import Path
from typing import List, Iterable, Union

IS_WINDOWS = sys.platform == 'win32'

which = 'where' if IS_WINDOWS else 'which'

SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

PathLike = Union[str, Path]


def lstr(li: Iterable[PathLike]) -> List[str]:
    return list(map(str, li))


def _is_cuda_file(path: str) -> bool:
    valid_ext = [
        '.cu',
        '.cuh',
    ]
    return Path(path).suffix in valid_ext
