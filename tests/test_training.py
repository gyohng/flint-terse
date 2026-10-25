import tempfile
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.flint import Model, ModelArgs
from mlx_lm.tuner.utils import linear_to_lora_layers

from corpus import chat_messages, check_answer, parse_assistant, select_trace, split_thinking, trace_cache, trace_scores
from export import write_hf
from train import loss_masks, make_loss, reasoning_loss, train_paired, training_attention
from batch_capture import request_sampler
from train_head import delta_norm, loss as head_loss
from train_prompt import adapt_prompt, prefix_cache, prompt_loss


def tiny_flint():
    model = Model(ModelArgs(
        model_type="flint", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        mlp_only_layers=[0], layer_types=["full_attention", "sliding_attention"],
        sliding_window=33, rms_norm_eps=1e-6, vocab_size=128,
    ))
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, m:
                hasattr(m, "to_quantized") and model.quant_predicate(p, m))
    model.freeze()
    linear_to_lora_layers(model, 2, {
        "rank": 2, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    })
    return model


class TrainingTests(unittest.TestCase):
    def test_paired_training_saves_and_restores_only_adapter_weights(self):
        model = tiny_flint()
        ids, offset = [2, 3, 6, 12, 7, 22, 1], 2
        hidden = model.model(mx.array([ids[:-1]]))[0, offset - 1:]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            features, corpus = folder / "features.safetensors", folder / "corpus.jsonl"
            mx.save_safetensors(str(features), {"hidden": hidden, "targets": mx.array(ids[offset:])}, metadata={"model": "tiny"})
            rows = [{"id": split, "split": split, "teacher_reasoning": "native", "assistant": {}} for split in ("train", "valid")]
            corpus.write_text("\n".join(json.dumps(r) for r in rows))
            args = SimpleNamespace(model="tiny", teacher_features=[str(features)] * 2, paired=corpus,
                max_tokens=32, iters=2, accumulate=1, seed=42, learning_rate=.0001, kl=2, start_weight=3, answer_weight=1, output=folder)
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
            with patch("train.training_tokens", return_value=(ids, offset)), patch("train.grad_checkpoint"), \
                 patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
                train_paired(model, tokenizer, args)
            saved = mx.load(str(folder / "adapters.safetensors"))
            current = dict(tree_flatten(model.trainable_parameters()))
            self.assertEqual(saved.keys(), current.keys())
            for key in saved:
                self.assertTrue(bool((saved[key] == current[key]).all()))

    def test_compact_thinking_loss_preserves_answers_by_probability(self):
        for closing in (1, 4):
            with self.subTest(closing=closing):
                logits = mx.random.normal((1, closing + 3, 16))
                targets = mx.arange(closing + 3)[None]
                reference = nn.log_softmax(logits[:, closing + 1:], axis=-1)
                value = reasoning_loss(logits, targets, closing, reference, 2, 3)
                changed = mx.concatenate((targets[:, :closing + 1], mx.full((1, 2), 15)), axis=1)
                self.assertEqual(value.item(), reasoning_loss(logits, changed, closing, reference, 2, 3).item())
                supervised = reasoning_loss(logits, targets, closing, reference, 2, 3, 1)
                expected = nn.losses.cross_entropy(logits[:, closing + 1:-1], targets[:, closing + 1:-1]).mean()
                self.assertAlmostEqual((supervised - value).item(), expected.item(), places=5)
                gradient = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 2, 3))(logits)
                self.assertLess(mx.max(mx.abs(gradient[:, closing + 1:])).item(), 1e-6)
                self.assertGreater(mx.max(mx.abs(gradient[:, :closing + 1])).item(), 0)

    def test_reserved_prompt_gradients_and_fusion_leave_other_rows_unchanged(self):
        model = tiny_flint()
        embedding, _ = adapt_prompt(model, [100, 101], [10, 11])
        prompt = [2, 3, 4, 5] * 10
        completion = mx.array([[6, 42, 7, 11, 1]])
        inputs = mx.array([[9, 100, 101, 3, 5, 8, 6, 42, 7, 11]])
        reference = nn.log_softmax(mx.random.normal((1, 2, 128)), axis=-1)
        full = mx.array([prompt + inputs[0].tolist()])
        with patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
            shared = prefix_cache(model, prompt[:20])
            cache = prefix_cache(model, prompt[20:], shared)
            self.assertEqual(shared[0].offset, 20)
            def whole(m):
                logits = m(full)[:, len(prompt) + 5:].astype(mx.float32)
                ce = nn.losses.cross_entropy(logits[:, :3], completion[:, :3]).mean()
                kl = (mx.exp(reference) * (reference - nn.log_softmax(logits[:, 3:], axis=-1))).sum(axis=-1).mean()
                return ce + kl
            expected, gradient = nn.value_and_grad(model, whole)(model)
            actual, cached = nn.value_and_grad(model, prompt_loss)(model, inputs, completion, cache, 5, 2, reference, 1)
            self.assertAlmostEqual(expected.item(), actual.item(), places=5)
            self.assertEqual([k for k, _ in tree_flatten(cached)], ["model.embed_tokens.lora_b"])
            self.assertLess(mx.max(mx.abs(gradient["model"]["embed_tokens"]["lora_b"] - cached["model"]["embed_tokens"]["lora_b"])).item(), 1e-5)
        original = embedding.embedding(mx.arange(128))
        fused = embedding.fuse(dequantize=True)
        unchanged = mx.array([i for i in range(128) if i not in (100, 101)])
        self.assertTrue(bool((fused.weight[unchanged] == original[unchanged]).all()))
        self.assertLess(mx.max(mx.abs(fused(mx.arange(128)) - embedding(mx.arange(128)))).item(), 1e-5)

    def test_head_training_matches_inference_and_keeps_base_frozen(self):
        from mlx_lm.tuner.lora import LoRALinear

        base = nn.Linear(64, 32, bias=False)
        base.weight = base.weight.astype(mx.float16)
        base = nn.QuantizedLinear.from_linear(base, group_size=32, bits=4)
        base.freeze()
        head = LoRALinear.from_base(base, r=2, scale=2, dropout=0)
        head.lora_b = mx.random.normal(head.lora_b.shape) * .02
        hidden = mx.random.normal((6, 64)).astype(mx.float16)
        targets = mx.arange(6)
        batch = {"hidden": hidden, "targets": targets, "weights": mx.ones((6,)),
                 "thought_weights": mx.ones((6,)), "answer_weights": mx.zeros((6,)),
                 "replay": mx.zeros((6,), dtype=mx.bool_)}
        ce = nn.losses.cross_entropy(head(hidden).astype(mx.float32), targets).mean()
        value, grad = nn.value_and_grad(head, head_loss)(head, batch, 0, 0, 0)
        self.assertAlmostEqual(value.item(), ce.item(), places=5)
        self.assertEqual({k for k, _ in tree_flatten(grad)}, {"lora_a", "lora_b"})
        self.assertAlmostEqual(delta_norm(head).item(), mx.linalg.norm(2 * head.lora_a @ head.lora_b).item(), places=5)
        batch["thought_weights"], batch["answer_weights"] = mx.zeros((6,)), mx.ones((6,))
        self.assertEqual(head_loss(head, batch, 0, 0, 0).item(), 0)
        self.assertGreater(head_loss(head, batch, 0, 1, 0).item(), 0)
        batch["answer_teacher_hidden"] = hidden + .3
        teacher = nn.log_softmax(head.linear(batch["answer_teacher_hidden"]).astype(mx.float32), axis=-1)
        student = nn.log_softmax(head(hidden).astype(mx.float32), axis=-1)
        expected = (mx.exp(teacher) * (teacher - student)).sum(axis=-1).mean()
        self.assertAlmostEqual(head_loss(head, batch, 0, 1, 0).item(), expected.item(), places=5)
        batch["replay"] = mx.ones((6,), dtype=mx.bool_)
        self.assertEqual(head_loss(head, batch, 0, 1, 0).item(), 0)
        self.assertGreater(head_loss(head, batch, 0, 0, 1).item(), 0)
        batch["teacher_hidden"] = hidden + .2
        batch["thought_weights"], batch["answer_weights"] = mx.ones((6,)), mx.zeros((6,))
        teacher = nn.log_softmax(head.linear(batch["teacher_hidden"]).astype(mx.float32), axis=-1)
        student = nn.log_softmax(head(hidden).astype(mx.float32), axis=-1)
        expected = (mx.exp(teacher) * (teacher - student)).sum(axis=-1).mean()
        self.assertAlmostEqual(head_loss(head, batch, 0, 0, 0).item(), expected.item(), places=5)
        batch["thought_weights"] = mx.zeros((6,))
        self.assertEqual(head_loss(head, batch, 0, 0, 0).item(), 0)

    def test_request_sampling_is_independent_of_other_tasks(self):
        logits = mx.log(mx.array([[.1, .15, .25, .3, .2] + [.001] * 20]))
        first, second, other = request_sampler(42, 1), request_sampler(42, 1), request_sampler(99, 1)
        for _ in range(10):
            expected = first(logits).item()
            for _ in range(3):
                other(logits).item()
            self.assertEqual(expected, second(logits).item())

    def test_completion_projection_preserves_loss_and_gradients(self):
        model = tiny_flint()
        for _, m in model.named_modules():
            if hasattr(m, "lora_b"):
                m.lora_b = mx.random.normal(m.lora_b.shape) * .01
        batch = mx.array([[2, 3, 6, 12, 7, 22, 1, 0], [2, 3, 4, 6, 7, 22, 1, 0]])
        lengths = mx.array([[2, 7], [3, 7]])
        loss = make_loss(model, 7, .1, "all")
        (whole, _), whole_grad = nn.value_and_grad(model, loss)(model, batch, lengths)
        (suffix, _), suffix_grad = nn.value_and_grad(model, loss)(model, batch, lengths, 1)
        self.assertAlmostEqual(whole.item(), suffix.item(), places=5)
        for (_, a), (_, b) in zip(tree_flatten(whole_grad), tree_flatten(suffix_grad)):
            self.assertLess(mx.max(mx.abs(a - b)).item(), 1e-5)

    def test_cached_likelihood_matches_full_forward_and_can_be_reused(self):
        model = tiny_flint()
        model.eval()
        tokens, offset = [2, 3, 4, 5, 8, 6, 42, 7, 11, 1], 5
        tokenizer = SimpleNamespace(encode=lambda *a, **k: tokens[:offset], convert_tokens_to_ids=lambda _: 7)
        nll = nn.losses.cross_entropy(model(mx.array([tokens[:-1]])).astype(mx.float32), mx.array([tokens[1:]]))[0]
        with patch("corpus.training_tokens", return_value=(tokens, offset)), patch("corpus.prompt_text", return_value="prompt"):
            cache = trace_cache(model, tokenizer, {})
            first = trace_scores(model, tokenizer, {}, cache)
            second = trace_scores(model, tokenizer, {}, cache)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["reasoning_nll"], nll[5].item(), places=5)
        self.assertAlmostEqual(first["answer_nll"], nll[7].item(), places=5)
        self.assertAlmostEqual(first["ending_nll"], nll[8].item(), places=5)

    def test_sampling_selects_by_checked_correctness_then_token_count(self):
        def trace(name, tokens, correct, complete=True):
            return dict(id=name, reasoning_tokens=tokens, output_tokens=tokens + 5,
                        complete=complete, correct=correct)
        wrong = trace("wrong", 1, False)
        truncated = trace("truncated", 2, True, False)
        long = trace("long", 80, True)
        short = trace("short", 12, True)
        unknown = trace("unverified", 3, None)
        self.assertIs(select_trace([wrong, truncated, long, short, unknown]), short)
        self.assertEqual(select_trace([trace("first", 50, None), unknown])["id"], "first")
        self.assertIs(select_trace([truncated, wrong]), wrong)

    def test_openclaw_grading_checks_structured_actions_and_json(self):
        tools = [{"function": {"name": "exec", "parameters": {
            "properties": {"command": {"type": "string"}}, "required": ["command"],
        }}}]
        row = {"tools": tools, "expected_tool": {"name": "exec", "arguments": {
            "command": "bash /tmp/calc.sh '100+232'",
        }}}
        for command, expected in [
            ('bash /tmp/calc.sh "100+232"', True),
            ('bash /tmp/calc.sh 100+233', False),
            ('bash /tmp/calc.sh 100+232; echo done', False),
        ]:
            with self.subTest(command=command):
                answer = f'<tool_call>exec<arg_key>command</arg_key><arg_value>{command}</arg_value></tool_call>'
                self.assertEqual(check_answer(row, answer, True), expected)
                self.assertFalse(check_answer(row, answer, False))
        for answer, expected in [('{"verified":false}', True), ('{"verified":0}', False),
                                 ('{"verified":true}', False), ('not json', False)]:
            with self.subTest(answer=answer):
                self.assertEqual(check_answer({"expected_json": {"verified": False}}, answer, True), expected)
        history = [{"role": "assistant", "tool_calls": [{"function": {
            "name": "exec", "arguments": {"command": "cat /tmp/input.txt"},
        }}]}, {"role": "tool", "content": "result"}]
        api = chat_messages(history)
        self.assertEqual(api[0]["tool_calls"][0]["id"], api[1]["tool_call_id"])
        self.assertIsInstance(api[0]["tool_calls"][0]["function"]["arguments"], str)
        self.assertIsInstance(history[0]["tool_calls"][0]["function"]["arguments"], dict)

    def test_hf_export_restores_experts_and_preserves_weights(self):
        from mlx_lm.utils import load_model

        model = tiny_flint()
        fused = [(n, m.fuse(dequantize=True)) for n, m in model.named_modules() if hasattr(m, "fuse")]
        model.update_modules(tree_unflatten(fused))
        model.eval()
        x = mx.array([[2, 6, 12, 7, 22]])
        expected = model(x)
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "hf"
            original = '{"tokenizer_class":"PreTrainedTokenizerFast"}'
            (Path(tmp) / "tokenizer_config.json").write_text(original)
            template = "<|reserved_150143|><|assistant|>"
            write_hf(model, asdict(model.args), tmp, output, shard_bytes=100_000, template=template)
            import json
            exported = json.loads((output / "tokenizer_config.json").read_text())
            self.assertEqual(exported["tokenizer_class"], "PreTrainedTokenizerFast")
            self.assertEqual(exported["chat_template"], template)
            self.assertEqual((output / "chat_template.jinja").read_text(), template)
            restored, _ = load_model(output)
            restored.eval()
            self.assertLess(mx.max(mx.abs(restored(x) - expected)).item(), 1e-4)

    def test_native_tool_call_arguments_remain_typed(self):
        tools = [{"function": {"name": "read", "parameters": {
            "properties": {"path": {"type": "string"}, "line": {"type": "integer"}},
            "required": ["path"],
        }}}]
        text = '<tool_call>read<arg_key>path</arg_key><arg_value>/tmp/a&b</arg_value><arg_key>line</arg_key><arg_value>3</arg_value></tool_call>'
        result = parse_assistant("Need file.", text, tools)
        self.assertEqual(result["tool_calls"][0]["function"]["arguments"], {"path": "/tmp/a&b", "line": 3})
        self.assertEqual(result["content"], "")

    def test_sink_attention_matches_fast_kernel_and_backpropagates(self):
        mx.random.seed(4)
        q = mx.random.normal((1, 4, 96, 32))
        k, v = mx.random.normal((2, 1, 2, 96, 32))
        sinks = mx.random.normal((4,))
        masks = ["causal", mx.arange(96)[None, :] <= mx.arange(96)[:, None]]
        for mask in masks:
            with self.subTest(mask=type(mask).__name__):
                fast = mx.fast.scaled_dot_product_attention(q, k, v, scale=32**-.5, mask=mask, sinks=sinks)
                eager = training_attention(q, k, v, None, 32**-.5, mask, sinks)
                self.assertLess(mx.max(mx.abs(fast - eager)).item(), 1e-5)
                grad = mx.grad(lambda x: training_attention(x, k, v, None, 32**-.5, mask, sinks).sum())(q)
                self.assertTrue(bool(mx.isfinite(grad).all()))

    def test_real_head_sink_gradients_match_explicit_attention(self):
        mx.random.seed(7)
        q = mx.random.normal((1, 32, 64, 128))
        k, v = mx.random.normal((2, 1, 4, 64, 128))
        sinks = mx.random.normal((32,))
        positions = mx.arange(64)
        window = (positions[:, None] >= positions[None, :]) & (positions[:, None] < positions[None, :] + 17)
        for mask in ("causal", window, mx.where(window, 0.0, -float("inf"))):
            with self.subTest(mask=type(mask).__name__):
                def reference(q, k, v):
                    logits = (q.reshape(1, 4, 8, 64, 128) * 128**-.5) @ k[:, :, None].swapaxes(-1, -2)
                    allowed = positions[:, None] >= positions[None, :] if isinstance(mask, str) else mask
                    logits = mx.where(allowed, logits, -float("inf")) if allowed.dtype == mx.bool_ else logits + allowed
                    sink = mx.broadcast_to(sinks.reshape(1, 4, 8, 1, 1), (*logits.shape[:-1], 1))
                    probs = mx.softmax(mx.concatenate((logits, sink), axis=-1), axis=-1)[..., :-1]
                    return (probs @ v[:, :, None]).reshape(q.shape)

                fused = lambda q, k, v: training_attention(q, k, v, None, 128**-.5, mask, sinks)
                self.assertLess(mx.max(mx.abs(reference(q, k, v) - fused(q, k, v))).item(), 1e-5)
                expected = mx.grad(lambda q, k, v: mx.square(reference(q, k, v)).mean(), argnums=(0, 1, 2))(q, k, v)
                actual = mx.grad(lambda q, k, v: mx.square(fused(q, k, v)).mean(), argnums=(0, 1, 2))(q, k, v)
                for a, b in zip(actual, expected):
                    self.assertLess((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item(), 1e-5)

    def test_masks_ignore_prompt_prior_thought_and_padding(self):
        batch = mx.array([[6, 7, 9, 6, 42, 7, 11, 1, 0, 0]])
        supervised, answer = loss_masks(batch, mx.array([[3, 8]]), 7)
        self.assertEqual(supervised.tolist(), [[False, False, True, True, True, True, True, False, False]])
        self.assertEqual(answer.tolist(), [[False, False, False, False, False, True, True, False, False]])

    def test_incomplete_and_wrong_answers_never_pass(self):
        for text, complete, expected in [
            ("<think>work</think>42", True, True),
            ("<think></think>42", True, True),
            ("<think>work", False, False),
            ("<think>work</think>142", True, False),
            ("<think>work</think>more work</think>42", True, False),
        ]:
            with self.subTest(text=text):
                _, answer, closed = split_thinking(text)
                self.assertEqual(check_answer({"expected": "42"}, answer, complete and closed), expected)

    def test_quantized_flint_gradients_and_fusion(self):
        mx.random.seed(3)
        model = tiny_flint()
        model.train()
        x, lengths = mx.array([[2, 3, 6, 12, 7, 22, 1, 0]]), mx.array([[2, 7]])
        adapters = [m for _, m in model.named_modules() if hasattr(m, "lora_b")]
        for m in adapters:
            m.lora_b = mx.random.normal(m.lora_b.shape) * 0.01
        loss = make_loss(model, 7, 0.1)
        (value, _), grad = nn.value_and_grad(model, loss)(model, x, lengths)
        plain, _ = make_loss(model, 7, 0)(model, x, lengths)
        mx.eval(value, plain, grad)
        self.assertGreater(value.item(), plain.item())
        self.assertTrue(all(m.scale == 2.0 for m in adapters))
        self.assertTrue(all("lora_" in k for k, _ in tree_flatten(grad)))
        self.assertTrue(all(bool(mx.isfinite(g).all()) for _, g in tree_flatten(grad)))
        self.assertTrue(any(bool((g != 0).any()) for _, g in tree_flatten(grad)))
        model.eval()
        expected = model(x)
        mx.eval(expected)
        fused = [(n, m.fuse(dequantize=True)) for n, m in model.named_modules() if hasattr(m, "fuse")]
        model.update_modules(tree_unflatten(fused))
        self.assertLess(mx.max(mx.abs(model(x) - expected)).item(), 1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            model.save_weights(str(path))
            self.assertTrue(path.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
