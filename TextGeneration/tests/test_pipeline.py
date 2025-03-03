#!/usr/bin/env python3
"""Tests for the full text generation pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from TextGeneration.utils.language_model import NGramLanguageModel
from TextGeneration.train import main_train
from TextGeneration.generate import main_generate


class TestPipeline(unittest.TestCase):
    """Test the full training and generation pipeline."""

    def test_full_pipeline(self):
        """Test the full training and generation pipeline."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            
            # Create sample training data
            training_folder = tmp_dir_path / "training_data"
            training_folder.mkdir()
            
            with open(training_folder / "sample1.txt", "w") as f:
                f.write("This is a sample text for testing n-gram models.\n")
                f.write("It has multiple lines with some repeated patterns.\n")
                f.write("The model should learn these patterns effectively.\n")
                f.write("This is how we test our language model.\n")
                f.write("Testing is important for any model we develop.\n")
            
            with open(training_folder / "sample2.txt", "w") as f:
                f.write("More sample text for testing our implementation.\n")
                f.write("This should work well with different inputs.\n")
                f.write("The model should be able to generate coherent text.\n")
                f.write("Testing with a variety of inputs helps validate the model.\n")
                f.write("We hope this model works effectively for all cases.\n")
            
            # Create training config
            model_path = tmp_dir_path / "trained_model.json"
            training_config = tmp_dir_path / "training_config.json"
            
            # Patch train.py to use lower min_count for testing
            original_train_func = NGramLanguageModel.train
            NGramLanguageModel.train = lambda self, text_lines, min_count=2: original_train_func(self, text_lines, min_count)
            
            with open(training_config, "w") as f:
                json.dump({
                    "trained_model": str(model_path),
                    "max_n_gram": 2,
                    "input_folder": str(training_folder)
                }, f)
            
            # Train model
            main_train(str(training_config))
            
            # Verify model was created
            self.assertTrue(model_path.exists(), "Model file was not created")
            
            # Restore original train function
            NGramLanguageModel.train = original_train_func
            
            # Create generation config
            output_path = tmp_dir_path / "output.json"
            generation_config = tmp_dir_path / "generation_config.json"
            
            with open(generation_config, "w") as f:
                json.dump({
                    "trained_model": str(model_path),
                    "max_n_gram": 2,
                    "texts": ["This is"],
                    "output_file": str(output_path),
                    "use_top_candidate": 1
                }, f)
            
            # Generate text
            main_generate(str(generation_config))
            
            # Verify output was created
            self.assertTrue(output_path.exists(), "Output file was not created")
            
            # Load generated output
            with open(output_path, "r") as f:
                output_data = json.load(f)
            
            # Verify output has generated texts
            self.assertIn("generated_texts", output_data)
            self.assertEqual(len(output_data["generated_texts"]), 1)
            self.assertTrue(output_data["generated_texts"][0].startswith("This is"))


if __name__ == "__main__":
    unittest.main()