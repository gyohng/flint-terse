import unittest

from code_checks import check_python
from corpus import check_answer, prefix_template
from make_corpus import arithmetic


class CorpusTests(unittest.TestCase):
    def test_learned_prefix_keeps_history_and_completion_boundary(self):
        from jinja2 import Environment

        template = """{%- for m in messages -%}
{%- if m.role == 'user' -%}<|user|>{{ m.content }}
{%- elif m.role == 'assistant' -%}
<|assistant|>{{ m.content }}
{%- endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{%- endif -%}"""
        messages = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Reply"},
                    {"role": "user", "content": "New"}]
        for role in ("user", "developer"):
            adapted = Environment().from_string(prefix_template(template, ["<|reserved_1|>"], role))
            prompt = adapted.render(messages=messages, add_generation_prompt=True)
            full = adapted.render(messages=messages + [{"role": "assistant", "content": "<think>notes</think>answer"}],
                                  add_generation_prompt=False)
            self.assertEqual(full, prompt + "<think>notes</think>answer")
            self.assertEqual(prompt.count("<|reserved_1|>"), 1)
            boundary = "<|developer|>" if role == "developer" else ""
            self.assertEqual(prompt, "<|user|>Old<|assistant|>Reply<|user|>New" + boundary + "<|reserved_1|><|assistant|>")

    def test_numeric_answers_require_the_whole_answer(self):
        for text, expected in [("42", True), ("42.0", True), ("41, not 42", False), ("142", False)]:
            self.assertEqual(check_answer({"expected_number": "42"}, text, True), expected)
        self.assertEqual(arithmetic("(48+24)/3"), 24)

    def test_python_checks_execute_tests_and_block_file_writes(self):
        self.assertTrue(check_python("def f(x): return x+1", ["assert f(2)==3"]))
        self.assertFalse(check_python("def f(x): return x-1", ["assert f(2)==3"]))
        self.assertFalse(check_python("pass", []))
        self.assertFalse(check_python("open('/tmp/flint-denied', 'w').write('x')", ["assert True"]))
