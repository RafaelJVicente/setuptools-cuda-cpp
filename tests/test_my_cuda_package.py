import unittest

from my_cuda_package import cuda_ext


class TestCudaext(unittest.TestCase):
    def test_hello(self):
        data = cuda_ext.hello()
        self.assertEqual(type(data), bytes)
        self.assertEqual(data, b'Hello CUDA!!!')


if __name__ == '__main__':
    unittest.main()
