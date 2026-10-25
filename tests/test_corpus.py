import unittest

from code_checks import check_python
from corpus import check_answer, prefix_template
from make_corpus import arithmetic
from preferences import add_preferences


class CorpusTests(unittest.TestCase):
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
