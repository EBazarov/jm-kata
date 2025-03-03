#!/usr/bin/env python3
"""Tests for the text preprocessor functionality."""

import unittest

from TextGeneration.utils.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    """Test cases for the Preprocessor class."""

    def test_clean_basic(self):
        """Test basic cleaning functionality."""
        self.assertEqual(Preprocessor.clean("Hello    world"), "Hello world")
        self.assertEqual(Preprocessor.clean("Hello_world"), "Hello world")
        self.assertEqual(Preprocessor.clean('Hello "world"'), "Hello world")
        self.assertEqual(Preprocessor.clean("Hello--world"), "Hello - world")

    def test_clean_edge_cases(self):
        """Test edge cases for cleaning."""
        self.assertEqual(Preprocessor.clean(""), "")
        self.assertEqual(Preprocessor.clean("!!!"), "!!!")
        self.assertEqual(Preprocessor.clean("  hello  "), "hello")
        self.assertEqual(Preprocessor.clean("test 123"), "test 123")
        self.assertEqual(Preprocessor.clean("test@example.com"), "test@example.com")


if __name__ == "__main__":
    unittest.main()
