#!/usr/bin/env python3
"""Tests for the file utility functions."""

import json
import tempfile
import unittest
from pathlib import Path

from TextGeneration.utils.files import json_to_schema, schema_to_json, read_dir
from TextGeneration.utils.schemas import OutputSchema, TrainingInputSchema


class TestFileUtils(unittest.TestCase):
    """Test cases for the file utility functions."""

    def test_schema_to_json(self):
        """Test converting a schema to JSON."""
        output_schema = OutputSchema(generated_texts=["test1", "test2"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"

            schema_to_json(file_path, output_schema)
            self.assertTrue(file_path.exists())

            with open(file_path, "r") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data, {"generated_texts": ["test1", "test2"]})

    def test_json_to_schema(self):
        """Test converting JSON to a schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"

            test_data = {
                "trained_model": "/path/to/model.json",
                "max_n_gram": 3,
                "input_folder": "/path/to/input",
            }

            with open(file_path, "w") as f:
                json.dump(test_data, f)

            schema = json_to_schema(str(file_path), TrainingInputSchema)
            self.assertEqual(schema.trained_model, Path("/path/to/model.json"))
            self.assertEqual(schema.max_n_gram, 3)
            self.assertEqual(schema.input_folder, Path("/path/to/input"))

    def test_read_dir(self):
        """Test reading lines from files in a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)

            with open(dir_path / "file1.txt", "w") as f:
                f.write("line1\nline2\n")

            with open(dir_path / "file2.txt", "w") as f:
                f.write("line3\nline4\n")

            with open(dir_path / "ignored.csv", "w") as f:
                f.write("should,be,ignored\n")

            lines = list(read_dir(dir_path))

            self.assertEqual(len(lines), 4)
            self.assertIn("line1\n", lines)
            self.assertIn("line2\n", lines)
            self.assertIn("line3\n", lines)
            self.assertIn("line4\n", lines)
            self.assertNotIn("should,be,ignored\n", lines)


if __name__ == "__main__":
    unittest.main()
