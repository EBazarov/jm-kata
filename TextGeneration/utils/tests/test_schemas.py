#!/usr/bin/env python3
"""Tests for the schema validation functionality."""

import unittest
from pathlib import Path
from pydantic import ValidationError

from TextGeneration.utils.schemas import TrainingInputSchema, InputSchema, OutputSchema


class TestSchemas(unittest.TestCase):
    """Test cases for the schema classes."""

    def test_valid_training_schema(self):
        """Test validation of valid training configuration."""
        valid_config = {
            "trained_model": "/path/to/model.json",
            "max_n_gram": 3,
            "input_folder": "/path/to/input",
        }

        schema = TrainingInputSchema(**valid_config)
        self.assertEqual(schema.trained_model, Path("/path/to/model.json"))
        self.assertEqual(schema.max_n_gram, 3)
        self.assertEqual(schema.input_folder, Path("/path/to/input"))

    def test_invalid_training_schema(self):
        """Test validation of invalid training configuration."""
        # Missing trained_model
        invalid_config1 = {"max_n_gram": 3, "input_folder": "/path/to/input"}

        # Missing input_folder
        invalid_config2 = {"trained_model": "/path/to/model.json", "max_n_gram": 3}

        # Invalid max_n_gram (not an integer)
        invalid_config3 = {
            "trained_model": "/path/to/model.json",
            "max_n_gram": "three",
            "input_folder": "/path/to/input",
        }

        with self.assertRaises(ValidationError):
            TrainingInputSchema(**invalid_config1)

        with self.assertRaises(ValidationError):
            TrainingInputSchema(**invalid_config2)

        with self.assertRaises(ValidationError):
            TrainingInputSchema(**invalid_config3)

    def test_valid_generation_schema(self):
        """Test validation of valid generation configuration."""
        valid_config = {
            "trained_model": "/path/to/model.json",
            "texts": ["This is a test"],
            "output_file": "/path/to/output.json",
            "max_n_gram": 3,
            "use_top_candidate": 5,
        }

        schema = InputSchema(**valid_config)
        self.assertEqual(schema.trained_model, Path("/path/to/model.json"))
        self.assertEqual(schema.max_n_gram, 3)
        self.assertEqual(schema.texts, ["This is a test"])
        self.assertEqual(schema.output_file, Path("/path/to/output.json"))
        self.assertEqual(schema.use_top_candidate, 5)

    def test_invalid_generation_schema(self):
        """Test validation of invalid generation configuration."""
        # Missing trained_model
        invalid_config1 = {
            "texts": ["This is a test"],
            "output_file": "/path/to/output.json",
            "max_n_gram": 3,
        }

        # Missing texts
        invalid_config2 = {
            "trained_model": "/path/to/model.json",
            "output_file": "/path/to/output.json",
            "max_n_gram": 3,
        }

        # Invalid texts (not a list)
        invalid_config3 = {
            "trained_model": "/path/to/model.json",
            "texts": "This is a test",
            "output_file": "/path/to/output.json",
            "max_n_gram": 3,
        }

        # Invalid use_top_candidate (not an integer)
        invalid_config4 = {
            "trained_model": "/path/to/model.json",
            "texts": ["This is a test"],
            "output_file": "/path/to/output.json",
            "max_n_gram": 3,
            "use_top_candidate": "five",
        }

        with self.assertRaises(ValidationError):
            InputSchema(**invalid_config1)

        with self.assertRaises(ValidationError):
            InputSchema(**invalid_config2)

        with self.assertRaises(ValidationError):
            InputSchema(**invalid_config3)

        with self.assertRaises(ValidationError):
            InputSchema(**invalid_config4)

    def test_output_schema(self):
        """Test output schema."""
        valid_output = {"generated_texts": ["Generated text 1", "Generated text 2"]}

        schema = OutputSchema(**valid_output)
        self.assertEqual(
            schema.generated_texts, ["Generated text 1", "Generated text 2"]
        )

        # Invalid output (missing generated_texts)
        with self.assertRaises(ValidationError):
            OutputSchema(**{})

        # Invalid output (generated_texts not a list)
        with self.assertRaises(ValidationError):
            OutputSchema(**{"generated_texts": "Not a list"})


if __name__ == "__main__":
    unittest.main()
