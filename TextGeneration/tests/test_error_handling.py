#!/usr/bin/env python3
"""Tests for error handling in the text generation code."""

import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from TextGeneration.utils.language_model import NGramLanguageModel
from TextGeneration.train import main_train
from TextGeneration.generate import main_generate


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in the text generation code."""

    def test_model_saving_nonexistent_directory(self):
        """Test error handling when saving to a nonexistent directory."""
        model = NGramLanguageModel(max_n_gram=2)
        model.train(text_lines=["This is a test"], min_count=1)
        
        # Create a path to a nonexistent directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            nonexistent_dir = Path(tmp_dir) / "nonexistent" / "directory" / "model.json"
            
            # The model should create the directory and save successfully
            model.save(file_path=nonexistent_dir)
            self.assertTrue(nonexistent_dir.exists())
    
    def test_model_loading_nonexistent_file(self):
        """Test error handling when loading a nonexistent model file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nonexistent_file = Path(tmp_dir) / "nonexistent.json"
            
            # Attempting to load a nonexistent file should raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                NGramLanguageModel.load(file_path=nonexistent_file)
    
    def test_model_loading_invalid_json(self):
        """Test error handling when loading an invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_json_file = Path(tmp_dir) / "invalid.json"
            
            # Create an invalid JSON file
            with open(invalid_json_file, 'w') as f:
                f.write("This is not valid JSON")
            
            # Attempting to load an invalid JSON file should raise ValueError
            with self.assertRaises(ValueError):
                NGramLanguageModel.load(file_path=invalid_json_file)
    
    def test_model_loading_invalid_structure(self):
        """Test error handling when loading a file with invalid model structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_structure_file = Path(tmp_dir) / "invalid_structure.json"
            
            # Create a JSON file with invalid model structure
            with open(invalid_structure_file, 'w') as f:
                json.dump({"not_a_model": True}, f)
            
            # The model should load but warn about missing fields
            model = NGramLanguageModel.load(file_path=invalid_structure_file)
            self.assertEqual(model.max_n_gram, 2)  # Default value
    
    def test_train_invalid_config(self):
        """Test error handling when training with an invalid configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_config_file = Path(tmp_dir) / "invalid_config.json"
            
            # Create an invalid configuration file
            with open(invalid_config_file, 'w') as f:
                f.write("This is not valid JSON")
            
            # Attempting to train with an invalid configuration should raise ValueError
            with self.assertRaises(ValueError):
                with patch('TextGeneration.train.train_logger'):  # Suppress logging
                    main_train(str(invalid_config_file))
    
    def test_generate_nonexistent_model(self):
        """Test error handling when generating with a nonexistent model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid configuration file with a nonexistent model
            config_file = Path(tmp_dir) / "config.json"
            nonexistent_model = Path(tmp_dir) / "nonexistent.json"
            output_file = Path(tmp_dir) / "output.json"
            
            config = {
                "trained_model": str(nonexistent_model),
                "max_n_gram": 2,
                "texts": ["This is a test"],
                "output_file": str(output_file),
                "use_top_candidate": 1
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Attempting to generate with a nonexistent model should raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                with patch('TextGeneration.generate.generate_logger'):  # Suppress logging
                    main_generate(str(config_file))


if __name__ == "__main__":
    unittest.main()