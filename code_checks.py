"""Run pure Python benchmark tests with no network or workspace file access."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path


def check_python(answer, tests, imports=()):
    if not tests:
        return False
    block = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    code = block[1] if block else answer
    source = "\n".join([*imports, code, *tests])
    allowed = {"math", "re", "collections", "itertools", "functools", "heapq", "bisect",
               "statistics", "fractions", "decimal", "string", "operator", "typing", "array", "copy"}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(n.name.split(".")[0] not in allowed for n in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] not in allowed):
            return False
    # Use the base interpreter: benchmark programs need only the standard library.
    python = str(Path(sys._base_executable).resolve())
    policy = '\n'.join([
        '(version 1)', '(allow default)', '(deny network*)', '(deny file-write*)',
        '(deny file-read-data (subpath "/Users") (subpath "/CODE") (subpath "/Volumes"))',
        '(deny process-fork)',
        f'(allow file-read-data (subpath {json.dumps(str(Path(sys.base_prefix).resolve()))}))',
    ])
    runner = "import resource,sys\nresource.setrlimit(resource.RLIMIT_CPU,(2,2))\nexec(compile(sys.stdin.read(),'<benchmark>','exec'))"
    try:
        result = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", policy, python, "-I", "-S", "-B", "-c", runner],
            input=source, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=4, cwd="/tmp",
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
