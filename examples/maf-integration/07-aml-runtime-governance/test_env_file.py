"""Tests for the demo's minimal .env loader."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import env_file
from env_file import load_env_file, parse_env_file

KEY = "AML_TEST_ENV_FILE_VALUE"


class EnvFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / ".env"
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop(KEY, None)
        env_file._LOADED.clear()

    def tearDown(self):
        self.env.stop()
        env_file._LOADED.clear()
        self.dir.cleanup()

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_parses_comments_quotes_export_and_inline_comments(self):
        self.write('# comment\n\nA=1\nexport B = two\nC="quoted # not a comment"\nD=x # trailing\nnot a line\n')
        self.assertEqual(
            parse_env_file(self.path),
            {"A": "1", "B": "two", "C": "quoted # not a comment", "D": "x"},
        )

    def test_missing_file_is_ignored(self):
        self.assertEqual(load_env_file(self.path), {})

    def test_terminal_value_takes_precedence(self):
        os.environ[KEY] = "from-terminal"
        self.write(f"{KEY}=from-file\n")
        self.assertEqual(load_env_file(self.path), {})
        self.assertEqual(os.environ[KEY], "from-terminal")

    def test_edits_apply_on_next_load_and_removal_restores_default(self):
        self.write(f"{KEY}=1024\n")
        load_env_file(self.path)
        self.assertEqual(os.environ[KEY], "1024")
        self.write(f"{KEY}=8192\n")
        load_env_file(self.path)
        self.assertEqual(os.environ[KEY], "8192")
        self.write("# removed\n")
        load_env_file(self.path)
        self.assertNotIn(KEY, os.environ)


if __name__ == "__main__":
    unittest.main()
