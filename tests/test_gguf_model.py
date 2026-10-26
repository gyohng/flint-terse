import unittest
from types import SimpleNamespace

import gguf
import mlx.core as mx
import numpy as np

from gguf_model import GGUFLinear, repack


class GGUFTests(unittest.TestCase):
    def test_exact_coefficients_and_input_gradients(self):
        rng = np.random.default_rng(42)
        for kind in (gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K,
                     gguf.GGMLQuantizationType.Q8_0):
            with self.subTest(kind=kind.name):
                block_size, block_bytes = gguf.GGML_QUANT_SIZES[kind]
                data = rng.integers(0, 256, (8 * 256 // block_size, block_bytes), dtype=np.uint8)
                scale = np.full((len(data), 1), .003, np.float16).view(np.uint8)
                if kind == gguf.GGMLQuantizationType.Q6_K:
                    data[:, -2:] = scale
                else:
                    data[:, :2] = scale
                    if kind in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q4_1):
                        data[:, 2:4] = scale
                tensor = SimpleNamespace(name='fixture', shape=np.array([256, 8]), tensor_type=kind, data=data)
                (w, s, b), bits, padded = repack(tensor)
                decoded = mx.dequantize(w, s, b, group_size=32, bits=bits)
                if padded:
                    decoded = decoded.reshape(8, -1, 32)[..., :16].reshape(8, 256)
                expected = mx.array(gguf.dequantize(data, kind).reshape(8, 256))
                self.assertTrue(mx.array_equal(decoded, expected).item())
                layer = GGUFLinear(tensor)
                x = mx.array(rng.normal(size=(2, 256)).astype(np.float32))
                np.testing.assert_allclose(np.array(layer(x)), np.array(x @ expected.T), rtol=2e-5, atol=2e-5)
                actual_grad = mx.grad(lambda v: layer(v).sum())(x)
                expected_grad = mx.grad(lambda v: (v @ expected.T).sum())(x)
                np.testing.assert_allclose(np.array(actual_grad), np.array(expected_grad), rtol=2e-5, atol=2e-5)
                tensor.shape = np.array([256, 4, 2])
                experts = GGUFLinear(tensor)
                indices = mx.array([1, 0])
                inputs = x[:, None, :]
                dense = expected.reshape(2, 4, 256).swapaxes(-1, -2)
                reference = lambda v: mx.gather_mm(v, dense, rhs_indices=indices)
                np.testing.assert_allclose(np.array(experts(inputs, indices)), np.array(reference(inputs)), rtol=2e-5, atol=2e-5)
                actual_grad = mx.grad(lambda v: experts(v, indices).sum())(inputs)
                expected_grad = mx.grad(lambda v: reference(v).sum())(inputs)
                np.testing.assert_allclose(np.array(actual_grad), np.array(expected_grad), rtol=2e-5, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
