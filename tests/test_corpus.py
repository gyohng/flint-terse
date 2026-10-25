import unittest

from code_checks import check_python
from corpus import check_answer
from make_corpus import arithmetic


class CorpusTests(unittest.TestCase):
    def test_numeric_answers_require_the_whole_answer(self):
        for text, expected in [("42", True), ("42.0", True), ("41, not 42", False), ("142", False)]:
            self.assertEqual(check_answer({"expected_number": "42"}, text, True), expected)
        self.assertEqual(arithmetic("(48+24)/3"), 24)

    def test_python_checks_execute_tests_and_block_file_writes(self):
        self.assertTrue(check_python("def f(x): return x+1", ["assert f(2)==3"]))
        self.assertFalse(check_python("def f(x): return x-1", ["assert f(2)==3"]))
        self.assertFalse(check_python("pass", []))
        self.assertFalse(check_python("open('/tmp/flint-denied', 'w').write('x')", ["assert True"]))
