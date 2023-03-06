from typing import List, Tuple, Union

from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetCudaComputeCapability


class NVML:
    def __enter__(self):
        nvmlInit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvmlShutdown()


def get_device_capability(device_number: int = None) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    arch_list: List[Tuple[int, int]] = []
    with NVML():
        device_count = nvmlDeviceGetCount()
        for device_index in range(device_count):
            device_ptr = nvmlDeviceGetHandleByIndex(device_index)
            arch_list.append(nvmlDeviceGetCudaComputeCapability(device_ptr))

        if device_number is None:
            return arch_list
        return arch_list[device_number]


def get_device_capability_str(device_number: int = None) -> Union[str, List[str]]:
    if device_number is None:
        return [f'{major}.{minor}' for major, minor in get_device_capability()]

    major, minor = get_device_capability(device_number)
    return f'{major}.{minor}'


def get_arch_list() -> List[str]:
    # TODO Check per cuda version supported architectures
    return []
