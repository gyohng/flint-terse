import json
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from code_checks import check_python
from corpus import capture, check_answer, generate_api, prefix_template
from finetune import accuracy_can_pass
from make_corpus import arithmetic
from preferences import add_preferences, repair_answers
from prune import editing_messages, join_passages, passages, reasoning_shift, selected_passages


class CorpusTests(unittest.TestCase):
    def test_accuracy_margin_covers_the_whole_selected_test_set(self):
        for total, baseline, done, correct, ungraded, expected in [
            (100, 86, 89, 74, 0, True),
            (100, 86, 89, 72, 0, False),
            (200, 172, 100, 82, 0, True),
            (200, 172, 200, 168, 0, True),
            (200, 172, 200, 167, 0, False),
            (100, 86, 100, 84, 20, True),
            (100, 86, 100, 83, 20, False),
        ]:
            base = [{'correct': x} for x in [True] * baseline + [False] * (total - baseline) + [None] * ungraded]
            candidate = [{'correct': x} for x in [True] * correct + [False] * (done - correct)]
            with self.subTest(total=total, done=done, correct=correct, ungraded=ungraded):
                self.assertEqual(accuracy_can_pass(base, candidate, 2), expected)

    def test_api_retry_keeps_the_same_request(self):
        args = SimpleNamespace(model='candidate', max_tokens=64, temperature=1, seed=42,
                               api='http://localhost:1234/v1', api_chat=False)
        error = urllib.error.HTTPError(args.api, 500, 'busy', {}, io.BytesIO(b'busy'))
        response = io.BytesIO(json.dumps({'choices': [{'text': 'answer', 'finish_reason': 'stop'}],
                                         'usage': {'completion_tokens': 1}}).encode())
        with patch('corpus.urllib.request.urlopen', side_effect=[error, response]) as request, \
             patch('corpus.time.sleep'):
            self.assertEqual(generate_api('prompt', args, {}, None)[:3], ('answer', 1, 'stop'))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0].data, request.call_args_list[1].args[0].data)

    def test_capture_resumes_without_repeating_saved_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'tasks.jsonl', Path(directory) / 'results.jsonl'
            source.write_text(''.join(json.dumps({'id': str(i), 'messages': [{'role': 'user', 'content': str(i)}],
                                                   'expected_number': str(i)}) + '\n' for i in (1, 2, 3)))
            args = SimpleNamespace(input=source, output=output, model='candidate', adapter=None,
                                   max_tokens=64, temperature=1, seed=42, samples=1, limit=1,
                                   api='http://localhost:1234/v1', api_chat=False, tokenizer='unused', resume=False)
            tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('corpus.prompt_text', return_value='prompt'), \
                 patch('corpus.generate_api', side_effect=[('<think>x</think>1', 3, 'stop', 0, None),
                                                          ('<think>x</think>2', 3, 'stop', 0, None),
                                                          ('<think>3</think>', 2, 'stop', 0, None)]) as generate:
                capture(args)
                prefix = output.read_bytes()
                args.resume, args.limit = True, None
                capture(args)
                self.assertEqual([call.args[2]['id'] for call in generate.call_args_list], ['1', '2', '3'])
                self.assertTrue(output.read_bytes().startswith(prefix))
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                self.assertFalse(rows[-1]['complete'])
                self.assertFalse(rows[-1]['correct'])
                args.seed = 43
                with self.assertRaises(ValueError):
                    capture(args)

    def test_pruning_preserves_code_and_selects_only_original_passages(self):
        code = '```python\nx = 1.25\n\nprint("Done. Next.")\n```'
        source = 'We need answer. Compute 2.5 * 4 = 10.\n\n' + code + '\n\nDone.'
        parts = passages(source)
        self.assertEqual(parts, ['We need answer.', 'Compute 2.5 * 4 = 10.', code, 'Done.'])
        self.assertEqual(selected_passages('```json\n[3, 2, 3]\n```', len(parts)), [1, 2])
        self.assertEqual(selected_passages('[]', len(parts)), [])
        self.assertEqual(selected_passages('["2", "3"]', len(parts)), [1, 2])
        self.assertEqual(join_passages(source, parts, range(len(parts))), source)
        self.assertEqual(join_passages(source, parts, [1, 3]), 'Compute 2.5 * 4 = 10.\n\nDone.')
        for invalid in ('[0]', '[5]', '[true]', '[1.5]', '{}', 'Use [2]'):
            with self.subTest(selection=invalid), self.assertRaises(ValueError):
                selected_passages(invalid, len(parts))

    def test_pruning_compares_the_same_retained_tokens_and_closing_tag(self):
        original = {"reasoning_ids": [1, 2, 3, 4], "reasoning_losses": [.01, .02, 2, 3], "thinking_end_nll": .1}
        edited = {"reasoning_ids": [3, 4], "reasoning_losses": [2.3, 3.3], "thinking_end_nll": .4}
        self.assertAlmostEqual(reasoning_shift(original, edited), .3)
        self.assertAlmostEqual(reasoning_shift(original, original), 0)
        self.assertAlmostEqual(reasoning_shift(original, {"reasoning_ids": [], "reasoning_losses": [], "thinking_end_nll": 4}), 3.9)
        self.assertEqual(reasoning_shift(original, {**edited, "reasoning_ids": [9, 10]}), float("inf"))

    def test_passage_editor_keeps_tool_schema_and_answer_as_data(self):
        row = {"messages": [{"role": "user", "content": "Search for a file"}],
               "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
               "assistant": {"reasoning_content": "Search first.", "content": "", "tool_calls": [{"function": {"name": "search"}}]}}
        data = json.loads(editing_messages(row, ["Search first."])[-1]["content"])
        self.assertEqual(data["tools"], row["tools"])
        self.assertEqual(data["task"], row["messages"])
        self.assertEqual(data["answer"]["tool_calls"], row["assistant"]["tool_calls"])
        self.assertNotIn("reasoning_content", data["answer"])

    def test_preferences_recheck_answers_and_exclude_incomplete_attempts(self):
        row = {"id": "sum", "messages": [{"role": "user", "content": "20+22?"}],
               "expected_number": 42, "approved": True, "assistant": {"content": "42"}}
        candidates = []
        for answer, complete in [("41", True), ("42", True), ("0", False)]:
            candidates.append({"id": "sum", "messages": row["messages"], "complete": complete,
                "text": f"<think>notes</think>{answer}", "assistant": {"content": answer},
                "ending": "<|user|>", "output_tokens": len(answer), "correct": True})
        paired = add_preferences([row], candidates)[0]
        self.assertEqual(paired["rejected_assistant"]["content"], "41")
        self.assertEqual(paired["assistant"], row["assistant"])
        self.assertNotIn("rejected_assistant", row)
        formatted = {**candidates[0], "text": "<think>notes</think>$42"}
        empty = {**candidates[0], "text": "<think>notes</think>"}
        self.assertNotIn("rejected_assistant", add_preferences([row], [formatted, empty])[0])
        with self.assertRaises(ValueError):
            add_preferences([{**row, "messages": []}], candidates)

    def test_answer_repairs_keep_verified_code_and_exclude_truncated_notes(self):
        row = {"id": "sum", "messages": [{"role": "user", "content": "Add one."}],
               "tests": ["assert f(2) == 3"], "approved": True, "teacher_reasoning": "native",
               "assistant": {"reasoning_content": "correct notes", "content": "def f(x): return x+1"}}
        for notes, finish, accepted in [("wrong notes", "stop", True), ("", "stop", True),
                                         ("unfinished", "length", False), ("<think>nested", "stop", False)]:
            with self.subTest(notes=notes, finish=finish):
                candidate = {"id": "sum", "messages": row["messages"], "finish_reason": finish,
                             "assistant": {"reasoning_content": notes, "content": ""}}
                result = repair_answers([row], [candidate])
                self.assertEqual(bool(result), accepted)
                if accepted:
                    self.assertEqual(result[0]["assistant"], {**row["assistant"], "reasoning_content": notes})
                    self.assertEqual(result[0]["teacher_reasoning"], "native")
                    self.assertTrue(result[0]["answer_only"])
        self.assertEqual(row["assistant"]["reasoning_content"], "correct notes")
        with self.assertRaises(ValueError):
            repair_answers([{**row, "assistant": {"content": "def f(x): return x-1"}}], [])
        with self.assertRaises(ValueError):
            repair_answers([row], [{**candidate, "messages": []}])

    def test_learned_prefix_keeps_history_and_completion_boundary(self):
        from jinja2 import Environment

        template = """{%- for m in messages -%}
{%- if m.role == 'system' -%}<|system|>{{ m.content }}
{%- elif m.role == 'user' -%}
<|user|>{{ m.content }}
{%- elif m.role == 'assistant' -%}
<|assistant|>{{ m.content }}
{%- endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{%- endif -%}"""
        messages = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Reply"},
                    {"role": "user", "content": "New"}]
        for role in ("user", "developer", "system"):
            adapted = Environment().from_string(prefix_template(template, ["<|reserved_1|>"], role))
            prompt = adapted.render(messages=messages, add_generation_prompt=True)
            full = adapted.render(messages=messages + [{"role": "assistant", "content": "<think>notes</think>answer"}],
                                  add_generation_prompt=False)
            self.assertEqual(full, prompt + "<think>notes</think>answer")
            self.assertEqual(prompt.count("<|reserved_1|>"), 1)
            if role == "system":
                expected = "<|system|><|reserved_1|><|user|>Old<|assistant|>Reply<|user|>New<|assistant|>"
            else:
                boundary = f"<|{role}|>" if role != "user" else ""
                expected = "<|user|>Old<|assistant|>Reply<|user|>New" + boundary + "<|reserved_1|><|assistant|>"
            self.assertEqual(prompt, expected)

    def test_numeric_answers_require_the_whole_answer(self):
        for text, expected in [("42", True), ("42.0", True), ("41, not 42", False), ("142", False)]:
            self.assertEqual(check_answer({"expected_number": "42"}, text, True), expected)
        self.assertEqual(arithmetic("(48+24)/3"), 24)

    def test_python_checks_execute_tests_and_block_file_writes(self):
        self.assertTrue(check_python("def f(x): return x+1", ["assert f(2)==3"]))
        self.assertFalse(check_python("def f(x): return x-1", ["assert f(2)==3"]))
        self.assertFalse(check_python("pass", []))
        self.assertFalse(check_python("open('/tmp/flint-denied', 'w').write('x')", ["assert True"]))
