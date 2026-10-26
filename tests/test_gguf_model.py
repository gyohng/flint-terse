import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gguf
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from safetensors.numpy import save_file

from export_adapter import main as export_adapter
from gguf_model import GGUFLinear, repack


class GGUFTests(unittest.TestCase):
    def test_export_preserves_embedding_and_projection_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, adapter = root / 'model', root / 'adapter'
            model.mkdir()
            adapter.mkdir()
            (model / 'config.json').write_text(json.dumps({'num_hidden_layers': 2}))
            (model / 'chat_template.jinja').write_text(r'{{ text | replace("\x00", "") }}')
            (adapter / 'adapter_config.json').write_text(json.dumps({'lora_parameters': {'rank': 2, 'scale': 2}}))
            fixtures = [('model.embed_tokens', 'token_embd.weight', 8, 4),
                        ('model.layers.1.self_attn.q_proj', 'blk.1.attn_q.weight', 4, 6),
                        ('model.layers.1.self_attn.v_proj', 'blk.1.attn_v.weight', 4, 2)]
            weights = {}
            for prefix, _, width, output in fixtures:
                weights[prefix + '.lora_a'] = np.arange(width * 2, dtype=np.float32).reshape(width, 2) / 16
                weights[prefix + '.lora_b'] = np.arange(2 * output, dtype=np.float32).reshape(2, output) / 16
            save_file(weights, adapter / 'adapters.safetensors')
            export_adapter(['--adapter', str(adapter), '--model', str(model)])
            reader = gguf.GGUFReader(adapter / 'adapter.gguf')
            self.assertEqual(reader.fields['adapter.lora.alpha'].contents(), 4)
            tensors = {t.name: t for t in reader.tensors}
            for prefix, mapped, width, _ in fixtures:
                a, b = weights[prefix + '.lora_a'], weights[prefix + '.lora_b']
                ga, gb = tensors[mapped + '.lora_a'], tensors[mapped + '.lora_b']
                self.assertEqual(ga.tensor_type, gguf.GGMLQuantizationType.F32)
                self.assertEqual(gb.tensor_type, gguf.GGMLQuantizationType.F32)
                if prefix == 'model.embed_tokens':
                    actual = ga.data[[1, 3]] @ gb.data.T * 2
                    expected = a[[1, 3]] @ b * 2
                else:
                    x = np.arange(width, dtype=np.float32)[None]
                    actual = (x @ ga.data.T) @ gb.data.T * 2
                    expected = (x @ a) @ b * 2
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual((adapter / 'chat_template.jinja').read_text(), '{{ text }}')

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
                adapter = layer.to_lora(r=2, scale=2, dropout=0)
                adapter.lora_b = mx.ones_like(adapter.lora_b) * .01
                dense_adapter = lambda v: v @ expected.T + 2 * (v @ adapter.lora_a) @ adapter.lora_b
                np.testing.assert_allclose(np.array(adapter(x)), np.array(dense_adapter(x)), rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(np.array(mx.grad(lambda v: adapter(v).sum())(x)),
                                           np.array(mx.grad(lambda v: dense_adapter(v).sum())(x)), rtol=2e-5, atol=2e-5)
                self.assertEqual({name for name, _ in tree_flatten(adapter.trainable_parameters())}, {'lora_a', 'lora_b'})
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
