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
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

from corpus import chat_messages, check_answer, configure_adapter_tokenizer, parse_assistant, select_trace, split_thinking, trace_cache, trace_scores
from export import write_hf
from train import completion_logp, freeze_thinking_embedding, frozen_hidden, loss_masks, main as train_main, make_loss, narration_loss, preference_loss, reasoning_loss, retention_loss, train_paired, training_attention
from batch_capture import request_sampler
from train_head import delta_norm, loss as head_loss
from train_prompt import adapt_prompt, prefix_cache, prompt_loss, main as prompt_main


def tiny_flint(targets=None, rank=2):
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
        "rank": rank, "scale": 2.0, "dropout": 0.0,
        "keys": targets or ["self_attn.q_proj", "self_attn.v_proj"],
    })
    return model


class TrainingTests(unittest.TestCase):
    def test_retention_emphasizes_answer_entry_without_locking_thinking_openings(self):
        logits = mx.zeros((1, 32, 4))
        reference = nn.log_softmax(logits, axis=-1)
        for closing in (4, 12):
            with self.subTest(closing=closing):
                self.assertEqual(retention_loss(logits, reference, closing).item(), 0)
                losses = [retention_loss(logits.at[:, position, 0].add(2), reference, closing).item()
                          for position in (1, closing + 1, 31)]
                self.assertGreater(losses[0], 0)
                self.assertAlmostEqual(losses[0], losses[2])
                self.assertGreater(losses[1], losses[0])

    def test_entry_supervision_and_answer_transition(self):
        logits = mx.zeros((1, 8, 32))
        targets = mx.array([[1, 4, 5, 6, 2, 7, 8, 3]])
        for closing in (1, 4):
            with self.subTest(closing=closing):
                reference = nn.log_softmax(logits[:, closing + 1:], axis=-1)
                grad = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 0, entry_only=True))(logits)
                self.assertLess(grad[0, 1, 4].item(), 0)
                self.assertGreater(grad[0, 1, 7].item(), 0)
                self.assertEqual(mx.max(mx.abs(grad[:, :1])).item(), 0)
                self.assertEqual(mx.max(mx.abs(grad[:, 2:])).item(), 0)
                baseline = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 0))(logits)
                weighted = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 0, entry_weight=2))(logits)
                answer = mx.grad(lambda x: nn.losses.cross_entropy(x[:, closing + 1:closing + 2],
                    targets[:, closing + 1:closing + 2]).mean())(logits)
                transition = mx.grad(lambda x: nn.losses.cross_entropy(x[:, closing:closing + 1],
                    targets[:, closing:closing + 1]).mean())(logits) if closing > 1 else 0
                self.assertLess(mx.max(mx.abs(weighted - baseline - 2 * (grad + transition + answer))).item(), 1e-6)

    def test_narration_penalty_changes_only_the_first_thinking_word(self):
        logits = mx.zeros((1, 12, 32))
        grad = mx.grad(lambda x: narration_loss(x, [3, 5]))(logits)
        self.assertGreater(grad[0, 1, 3].item(), 0)
        self.assertGreater(grad[0, 1, 5].item(), 0)
        self.assertLess(grad[0, 1, 7].item(), 0)
        self.assertEqual(mx.max(mx.abs(grad[:, :1])).item(), 0)
        self.assertEqual(mx.max(mx.abs(grad[:, 2:])).item(), 0)
        lowered = logits.at[:, 1, 3].add(-1).at[:, 1, 5].add(-1)
        self.assertLess(narration_loss(lowered, [3, 5]).item(), narration_loss(logits, [3, 5]).item())

    def test_opening_training_keeps_attention_and_uses_same_prefix_retention(self):
        model = tiny_flint()
        for _, module in model.named_modules():
            if hasattr(module, 'lora_b'):
                module.lora_b = mx.random.normal(module.lora_b.shape) * .01
        weights = dict(tree_flatten(model.trainable_parameters()))
        ids, offset = [2, 3, 5, 6, 42, 7, 11, 1], 3
        hidden = model.model(mx.array([ids[:-1]]))[0, offset - 1:]
        mx.eval(weights, hidden)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output = folder / 'source', folder / 'output'
            source.mkdir()
            config = {'model': 'tiny', 'num_layers': 2, 'base_model': {'sha256': 'tiny'},
                      'lora_parameters': {'rank': 2, 'scale': 2., 'dropout': 0.,
                                          'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}
            (source / 'adapter_config.json').write_text(json.dumps(config))
            features, corpus = folder / 'features.safetensors', folder / 'corpus.jsonl'
            mx.save_safetensors(str(features), {'hidden': mx.concatenate([hidden] * 3),
                'targets': mx.array(ids[offset:] * 3)}, metadata={'model': 'tiny', 'adapter': str(source)})
            corpus.write_text('\n'.join(json.dumps(row) for row in [
                {'id': 'style', 'split': 'train'}, {'id': 'valid', 'split': 'valid'},
                {'id': 'retention', 'split': 'train', 'preserve_only': True}]))
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda t:
                {'<think>': 6, '</think>': 7, '<|assistant|>': 5, '<|user|>': 2}[t])
            with patch('train_prompt.training_tokens', return_value=(ids, offset)):
                prompt_main([str(corpus), '--model', 'tiny', '--teacher-features', str(features),
                    '--attention-adapter', str(source), '--think-opening', '--steps', '1',
                    '--accumulate', '2', '--output', str(output)], loaded=(model, tokenizer))
            saved = mx.load(str(output / 'adapters.safetensors'))
            for key in weights:
                self.assertTrue(bool((saved[key] == weights[key]).all()), key)
            embedding = saved['model.embed_tokens.lora_b']
            self.assertTrue(bool((embedding[0] != 0).any()))
            self.assertTrue(bool((embedding[1] == 0).all()))
            retained = json.loads((output / 'adapter_config.json').read_text())
            self.assertEqual(retained['lora_parameters']['scale'], 2.)
            self.assertEqual(retained['num_layers'], 2)

    def test_frozen_layer_cache_preserves_likelihood_and_gradients(self):
        mx.random.seed(19)
        model = tiny_flint()
        model.layers[0].freeze()
        for _, module in model.layers[1].named_modules():
            if hasattr(module, "lora_b"):
                module.lora_b = mx.random.normal(module.lora_b.shape) * .01
        model.train()
        with patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
            for length in (9, 65):
                ids = (mx.arange(length) % 100).tolist()
                cached = frozen_hidden(model, ids, 1)
                full = nn.value_and_grad(model, lambda m: completion_logp(m, ids, 3))(model)
                reused = nn.value_and_grad(model, lambda m: completion_logp(m, ids, 3, cached, 1))(model)
                self.assertAlmostEqual(full[0].item(), reused[0].item(), places=5)
                expected, actual = dict(tree_flatten(full[1])), dict(tree_flatten(reused[1]))
                self.assertEqual(expected.keys(), actual.keys())
                self.assertTrue(any(bool((g != 0).any()) for g in expected.values()))
                for name, grad in expected.items():
                    self.assertLess(mx.max(mx.abs(grad - actual[name])).item(), 1e-5)

    def test_correctness_preferences_are_centered_on_the_starting_policy(self):
        chosen, rejected = mx.array(-20.0), mx.array(-5.0)
        reference = -15.0
        initial = preference_loss(chosen, rejected, reference, .1).item()
        self.assertAlmostEqual(initial, .69314718, places=6)
        self.assertLess(preference_loss(chosen + 1, rejected, reference, .1).item(), initial)
        self.assertLess(preference_loss(chosen, rejected - 1, reference, .1).item(), initial)
        grad_chosen = mx.grad(lambda x: preference_loss(x, rejected, reference, .1))(chosen)
        grad_rejected = mx.grad(lambda x: preference_loss(chosen, x, reference, .1))(rejected)
        self.assertLess(grad_chosen.item(), 0)
        self.assertGreater(grad_rejected.item(), 0)

    def test_fixed_thinking_tokens_survive_attention_checkpoint_roundtrip(self):
        mx.random.seed(31)
        original = tiny_flint()
        embedding, _ = adapt_prompt(original, [6, 7], [6, 7])
        embedding.lora_b = mx.random.normal(embedding.lora_b.shape) * .01
        original.eval()
        weights = {"model.embed_tokens." + k: getattr(embedding, k) for k in ("lora_a", "lora_b")}
        mx.random.seed(31)
        combined = tiny_flint(rank=4)
        config = {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": ["model.embed_tokens"]}
        linear_to_lora_layers(combined, 2, config)
        frozen = freeze_thinking_embedding(combined.model.embed_tokens, weights, 1.0)
        combined.eval()
        inputs = mx.array([[2, 6, 12, 7, 22]])
        self.assertLess(mx.max(mx.abs(original(inputs) - combined(inputs))).item(), 1e-5)
        self.assertFalse(any("embed_tokens" in k for k, _ in tree_flatten(combined.trainable_parameters())))
        with tempfile.TemporaryDirectory() as folder:
            saved = dict(tree_flatten(combined.trainable_parameters()))
            mx.save_safetensors(str(Path(folder) / "adapters.safetensors"), {**saved, **frozen})
            config["keys"] += ["self_attn.q_proj", "self_attn.v_proj"]
            (Path(folder) / "adapter_config.json").write_text(json.dumps({"num_layers": 2, "lora_parameters": config}))
            mx.random.seed(31)
            restored = tiny_flint(rank=4)
            # Remove its zero attention adapters before loading the combined checkpoint.
            from mlx_lm.tuner.utils import remove_lora_layers
            remove_lora_layers(restored)
            load_adapters(restored, folder)
            restored.eval()
            self.assertLess(mx.max(mx.abs(combined(inputs) - restored(inputs))).item(), 1e-5)

    def test_adding_adapter_targets_preserves_checkpoint_and_enables_mlp_gradients(self):
        targets = ["self_attn." + p + "_proj" for p in ("q", "k", "v", "o")]
        targets += ["mlp.shared_expert." + p + "_proj" for p in ("gate", "up", "down")]
        mx.random.seed(17)
        previous = tiny_flint()
        for _, m in previous.named_modules():
            if hasattr(m, "lora_b"):
                m.lora_b = mx.random.normal(m.lora_b.shape) * .01
        mx.random.seed(17)
        expanded = tiny_flint(targets)
        expanded.load_weights(tree_flatten(previous.trainable_parameters()), strict=False)
        inputs = mx.array([[2, 3, 6, 12, 7, 22, 1]])
        previous.eval()
        expanded.eval()
        self.assertLess(mx.max(mx.abs(previous(inputs) - expanded(inputs))).item(), 1e-5)
        expanded.train()
        with patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
            _, grads = nn.value_and_grad(expanded, lambda m: m(inputs).mean())(expanded)
            mlp = [g for key, g in tree_flatten(grads) if "shared_expert" in key and key.endswith("lora_b")]
            self.assertEqual(len(mlp), 3)
            self.assertTrue(all(bool(mx.isfinite(g).all()) and bool((g != 0).any()) for g in mlp))

    def test_paired_training_saves_and_restores_only_adapter_weights(self):
        model = tiny_flint()
        ids, offset = [2, 3, 6, 12, 7, 22, 1], 2
        original_ids = [2, 3, 6, 12, 13, 14, 7, 22, 1]
        rejected_ids = [2, 3, 6, 13, 7, 23, 1]
        initial_ratio = (completion_logp(model, ids, offset) - completion_logp(model, rejected_ids, offset)).item()
        initial_compact_ratio = (completion_logp(model, ids, offset) - completion_logp(model, original_ids, offset)).item()
        hidden = model.model(mx.array([original_ids[:-1]]))[0, offset - 1:]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            features, corpus = folder / "features.safetensors", folder / "corpus.jsonl"
            mx.save_safetensors(str(features), {"hidden": hidden, "targets": mx.array(original_ids[offset:])}, metadata={"model": "tiny", "adapter": "teacher"})
            rows = [{"id": split, "split": split, "teacher_reasoning": "native", "assistant": {}} for split in ("train", "valid")]
            rows[0].update(rejected_assistant={"content": "wrong"}, rejected_ending="<|user|>")
            rows.append({"id": "retention", "split": "train", "preserve_only": True, "assistant": {}})
            corpus.write_text("\n".join(json.dumps(r) for r in rows))
            args = SimpleNamespace(model="tiny", teacher_features=[str(features)] * 2, paired=[corpus],
                max_tokens=32, iters=2, accumulate=1, seed=42, learning_rate=.0001, kl=2, start_weight=3, answer_weight=1,
                opening_only=True, answer_only=False, resume=None, teacher_adapter='teacher', preference_weight=1, compact_preference_weight=1, preference_beta=.1, output=folder)
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
            def tokens(row, *a, **kw):
                if row["assistant"].get("reasoning_content") == "native":
                    return original_ids, offset
                return (rejected_ids if row["assistant"].get("content") == "wrong" else ids), offset
            with patch("train.training_tokens", side_effect=tokens), patch("train.grad_checkpoint"), \
                 patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
                train_paired(model, tokenizer, args)
                args.teacher_adapter = 'wrong-teacher'
                with self.assertRaisesRegex(ValueError, 'selected attention adapter'):
                    train_paired(model, tokenizer, args)
            saved = mx.load(str(folder / "adapters.safetensors"))
            current = dict(tree_flatten(model.trainable_parameters()))
            self.assertEqual(saved.keys(), current.keys())
            for key in saved:
                self.assertTrue(bool((saved[key] == current[key]).all()))
            final_ratio = (completion_logp(model, ids, offset) - completion_logp(model, rejected_ids, offset)).item()
            self.assertGreater(final_ratio, initial_ratio)
            final_compact_ratio = (completion_logp(model, ids, offset) - completion_logp(model, original_ids, offset)).item()
            self.assertGreater(final_compact_ratio, initial_compact_ratio)

    def test_opening_supervision_leaves_later_reasoning_free(self):
        gradients = []
        for closing in (1, 20):
            logits = mx.zeros((1, closing + 3, 32))
            targets = mx.zeros((1, closing + 3), dtype=mx.int32)
            reference = nn.log_softmax(logits[:, closing + 1:], axis=-1)
            grad = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 2, opening_only=True))(logits)
            gradients.append(grad[0, 1, 0].item())
            self.assertLess(grad[0, 1, 0].item(), 0)
            if closing > 16:
                self.assertLess(mx.max(mx.abs(grad[:, 17:])).item(), 1e-6)
        self.assertAlmostEqual(*gradients, places=6)

    def test_same_prefix_attention_teacher_handles_style_and_retention_rows(self):
        model = tiny_flint()
        ids, offset = [2, 3, 6, 12, 7, 22, 1], 2
        hidden = model.model(mx.array([ids[:-1]]))[0, offset - 1:]
        initial = completion_logp(model, ids, offset).item()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            features, corpus = folder / 'features.safetensors', folder / 'corpus.jsonl'
            mx.save_safetensors(str(features), {'hidden': mx.concatenate([hidden] * 3),
                'targets': mx.array(ids[offset:] * 3)}, metadata={'model': 'tiny', 'adapter': 'starting'})
            corpus.write_text('\n'.join(json.dumps(row) for row in [
                {'id': 'train', 'split': 'train'}, {'id': 'valid', 'split': 'valid'},
                {'id': 'retain', 'split': 'train', 'preserve_only': True}]))
            args = SimpleNamespace(model='tiny', teacher_features=[str(features)], paired=[corpus],
                resume='later-checkpoint', teacher_adapter='starting', same_prefix_teacher=True, max_tokens=32, iters=2, accumulate=1,
                seed=42, learning_rate=.0001, kl=2, start_weight=3, answer_weight=1,
                opening_only=False, answer_only=False, preference_weight=0, narration_penalty=1,
                compact_preference_weight=0, preference_beta=.1, output=folder)
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 7,
                encode=lambda word, **kw: [{'We': 3, 'The': 5, 'Okay': 8, 'Hmm': 9, 'I': 10}[word]])
            with patch('train.training_tokens', return_value=(ids, offset)), \
                 patch('train.grad_checkpoint'), patch('mlx_lm.models.flint.scaled_dot_product_attention', training_attention):
                train_paired(model, tokenizer, args)
                self.assertGreater(completion_logp(model, ids, offset).item(), initial)
                args.teacher_adapter = 'wrong-teacher'
                with self.assertRaisesRegex(ValueError, 'selected attention adapter'):
                    train_paired(model, tokenizer, args)

    def test_retention_rows_still_learn_correctness_preferences(self):
        model = tiny_flint()
        chosen, rejected, offset = [2, 3, 6, 12, 7, 22, 1], [2, 3, 6, 12, 7, 23, 1], 2
        def ratio():
            return (completion_logp(model, chosen, offset) - completion_logp(model, rejected, offset)).item()
        initial = ratio()
        hidden = model.model(mx.array([chosen[:-1]]))[0, offset - 1:]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            features, corpus = folder / 'features.safetensors', folder / 'corpus.jsonl'
            mx.save_safetensors(str(features), {'hidden': mx.concatenate([hidden] * 2),
                'targets': mx.array(chosen[offset:] * 2)}, metadata={'model': 'tiny', 'adapter': 'starting'})
            corpus.write_text('\n'.join(json.dumps({'id': split, 'split': split, 'preserve_only': True,
                'assistant': {'content': 'correct'}, 'rejected_assistant': {'content': 'wrong'},
                'rejected_ending': '<|user|>'}) for split in ('train', 'valid')))
            args = SimpleNamespace(model='tiny', teacher_features=[str(features)], paired=[corpus],
                resume='starting', same_prefix_teacher=True, max_tokens=32, iters=1, accumulate=1,
                seed=42, learning_rate=.001, kl=0, start_weight=3, answer_weight=0,
                opening_only=False, answer_only=False, preference_weight=1,
                compact_preference_weight=0, preference_beta=.1, output=folder)
            def tokens(row, *a, **kw):
                return (rejected if row['assistant']['content'] == 'wrong' else chosen), offset
            with patch('train.training_tokens', side_effect=tokens), patch('train.grad_checkpoint'), \
                 patch('mlx_lm.models.flint.scaled_dot_product_attention', training_attention):
                train_paired(model, SimpleNamespace(convert_tokens_to_ids=lambda _: 7), args)
            self.assertGreater(ratio(), initial)

    def test_retention_uses_the_starting_policy_without_native_teacher_features(self):
        model = tiny_flint()
        ids, offset = [2, 3, 6, 12, 7, 22, 1], 2
        initial = completion_logp(model, ids, offset).item()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            features, corpus = folder / "features.safetensors", folder / "corpus.jsonl"
            mx.save_safetensors(str(features), {"hidden": mx.zeros((0, 64)), "targets": mx.array([], dtype=mx.int32)},
                                metadata={"model": "tiny"})
            corpus.write_text("\n".join(json.dumps({"id": split, "split": split, "preserve_only": True})
                                        for split in ("train", "valid")))
            args = SimpleNamespace(model="tiny", teacher_features=[str(features)], paired=[corpus],
                max_tokens=32, iters=1, accumulate=1, seed=42, learning_rate=.0001, kl=2, start_weight=3,
                answer_weight=1, opening_only=False, answer_only=True, preference_weight=0,
                compact_preference_weight=0, preference_beta=.1, output=folder)
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
            with patch("train.training_tokens", return_value=(ids, offset)), \
                 patch("train.grad_checkpoint"), patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
                train_paired(model, tokenizer, args)
            self.assertAlmostEqual(completion_logp(model, ids, offset).item(), initial, delta=1e-4)

    def test_resuming_keeps_saved_scale_and_frozen_adapter_layers(self):
        from mlx_lm.tuner.utils import remove_lora_layers

        model = tiny_flint()
        for _, module in model.named_modules():
            if hasattr(module, 'lora_b'):
                module.lora_b = mx.random.normal(module.lora_b.shape) * .01
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(weights)
        remove_lora_layers(model)
        ids, offset = [2, 3, 6, 12, 7, 22, 1], 2
        hidden = model.model(mx.array([ids[:-1]]))[0, offset - 1:]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'source'
            source.mkdir()
            mx.save_safetensors(str(source / 'adapters.safetensors'), weights)
            (source / 'adapter_config.json').write_text(json.dumps({'model': 'tiny', 'num_layers': 2,
                'lora_parameters': {'rank': 2, 'scale': .7, 'dropout': 0,
                                    'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}))
            features, corpus = directory / 'features.safetensors', directory / 'corpus.jsonl'
            mx.save_safetensors(str(features), {'hidden': mx.concatenate([hidden, hidden]),
                'targets': mx.array(ids[offset:] * 2)}, metadata={'model': 'tiny'})
            corpus.write_text('\n'.join(json.dumps({'id': split, 'split': split, 'assistant': {}, 'teacher_reasoning': 'native'})
                                        for split in ('train', 'valid')))
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
            for rank in (2, 4):
                with self.subTest(rank=rank):
                    output = directory / f'output-{rank}'
                    with patch('train.training_tokens', return_value=(ids, offset)), patch('train.grad_checkpoint'):
                        train_main(['--model', 'tiny', '--paired', str(corpus), '--teacher-features', str(features),
                            '--resume', str(source), '--layers', '2', '--train-layers', '1', '--rank', str(rank),
                            '--iters', '2', '--accumulate', '1', '--output', str(output)], loaded=(model, tokenizer))
                    saved = mx.load(str(output / 'adapters.safetensors'))
                    self.assertEqual(saved.keys(), weights.keys())
                    for name, original in weights.items():
                        old = saved[name][:, :2] if name.endswith('.lora_a') else saved[name][:2]
                        if rank > 2 or name.startswith('model.layers.0.'):
                            self.assertTrue(bool((old == original).all()), name)
                        if rank > 2 and name.endswith('.lora_b'):
                            changed = bool((saved[name][2:] != 0).any())
                            self.assertEqual(changed, name.startswith('model.layers.1.'), name)
                    self.assertEqual(json.loads((output / 'adapter_config.json').read_text())['lora_parameters']['scale'], .7)
                    remove_lora_layers(model)

    def test_answer_repair_does_not_supervise_sampled_thought_tokens(self):
        for closing in (1, 20):
            with self.subTest(closing=closing):
                logits = mx.zeros((1, closing + 4, 32))
                targets = mx.zeros((1, closing + 4), dtype=mx.int32)
                reference = nn.log_softmax(logits[:, closing + 1:], axis=-1)
                grad = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 2,
                    answer_weight=1, answer_only=True))(logits)
                self.assertEqual(mx.max(mx.abs(grad[:, :closing + 1])).item(), 0)
                self.assertLess(grad[0, closing + 1, 0].item(), 0)
                self.assertLess(mx.max(mx.abs(grad[:, -1])).item(), 1e-6)

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
                gradient = mx.grad(lambda x: reasoning_loss(x, targets, closing, reference, 2, 3, 1))(logits)
                self.assertLess(mx.max(mx.abs(gradient[:, -1])).item(), 1e-6)
                self.assertGreater(mx.max(mx.abs(gradient[:, closing + 1])).item(), 0)

    def test_long_replies_keep_a_strong_language_and_format_anchor(self):
        logits = mx.zeros((1, 35, 16))
        targets = mx.zeros((1, 35), dtype=mx.int32)
        reference = nn.log_softmax(mx.zeros((1, 32, 16)).at[:, :, 0].add(1), axis=-1)
        grad = mx.grad(lambda x: reasoning_loss(x, targets, 2, reference, 1))(logits)
        self.assertGreater(abs(grad[0, 3, 0].item()), abs(grad[0, 15, 0].item()))
        reference = nn.log_softmax(logits[:, 3:], axis=-1)
        grad = mx.grad(lambda x: reasoning_loss(x, targets, 2, reference, 1, answer_weight=1))(logits)
        self.assertGreater(abs(grad[0, 3, 0].item()), abs(grad[0, 15, 0].item()))
        self.assertLess(mx.max(mx.abs(grad[:, -1])).item(), 1e-6)

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

    def test_native_thinking_embedding_gradients_and_fusion_preserve_untuned_rows(self):
        for token_ids in ([6], [6, 7]):
            model = tiny_flint()
            embedding, initial = adapt_prompt(model, token_ids, token_ids)
            self.assertTrue(bool((embedding.lora_b == 0).all()))
            prompt = [2, 3, 4] * 5
            inputs, targets = mx.array([[5, 6, 42, 7, 11]]), mx.array([[6, 42, 7, 11, 1]])
            model.train()
            with patch("mlx_lm.models.flint.scaled_dot_product_attention", training_attention):
                cache = prefix_cache(model, prompt)
                reference = nn.log_softmax(model(mx.array([prompt + inputs[0].tolist()]))[:, len(prompt) + 3:], axis=-1)
                _, grads = nn.value_and_grad(model, prompt_loss)(model, inputs, targets, cache, 0, 2, reference, 2, 3, 1)
                change = grads["model"]["embed_tokens"]["lora_b"]
                self.assertTrue(bool((mx.linalg.norm(change, axis=-1) > 0).all()))
                embedding.lora_b = -.1 * change
                fused = embedding.fuse(dequantize=True)
                other = mx.array([i for i in range(128) if i not in token_ids])
                self.assertTrue(bool((fused(other) == embedding.embedding(other)).all()))
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "adapter_config.json").write_text(json.dumps({"think_tokens": True, "prompt_tokens": []}))
            tokenizer = SimpleNamespace(chat_template="original")
            self.assertIsNone(configure_adapter_tokenizer(tokenizer, folder))
            self.assertEqual(tokenizer.chat_template, "original")

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
        for answer in ([11], list(range(12, 24))):
            tokens, offset = [2, 3, 4, 5, 8, 6, 42, 7] + answer + [1], 5
            tokenizer = SimpleNamespace(encode=lambda *a, **k: tokens[:offset], convert_tokens_to_ids=lambda _: 7)
            nll = nn.losses.cross_entropy(model(mx.array([tokens[:-1]])).astype(mx.float32), mx.array([tokens[1:]]))[0]
            with patch("corpus.training_tokens", return_value=(tokens, offset)), patch("corpus.prompt_text", return_value="prompt"):
                cache = trace_cache(model, tokenizer, {})
                first = trace_scores(model, tokenizer, {}, cache, per_token=True)
                second = trace_scores(model, tokenizer, {}, cache, per_token=True)
            self.assertEqual(first, second)
            self.assertAlmostEqual(first["reasoning_nll"], nll[5].item(), places=5)
            self.assertEqual(first["reasoning_ids"], [42])
            self.assertAlmostEqual(first["reasoning_losses"][0], nll[5].item(), places=5)
            self.assertAlmostEqual(first["thinking_end_nll"], nll[6].item(), places=5)
            self.assertAlmostEqual(first["answer_nll"], nll[7:-1].mean().item(), places=5)
            self.assertAlmostEqual(first["answer_start_nll"], nll[7:min(15, len(nll) - 1)].mean().item(), places=5)
            self.assertAlmostEqual(first["ending_nll"], nll[-1].item(), places=5)

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
