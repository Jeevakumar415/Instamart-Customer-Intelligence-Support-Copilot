"""Syntax-only check for the generator; this never executes its module body."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class GenerateDataSyntaxTests(unittest.TestCase):
    def test_generator_parses_without_execution(self) -> None:
        source_path = Path(__file__).with_name("generate_data.py")
        ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


if __name__ == "__main__":
    unittest.main()
