#!/usr/bin/env python3
"""Tests for the n-gram language model functionality."""

import unittest
import tempfile
import json
from pathlib import Path

from TextGeneration.utils.language_model import NGramLanguageModel
from TextGeneration.utils.preprocessor import Preprocessor


class TestNGramLanguageModel(unittest.TestCase):
    """Test cases for the NGramLanguageModel class."""

    def test_basic_functionality(self):
        """Test basic functionality of the NGramLanguageModel class."""
        training_lines = [
            "this is a test sentence",
            "this is another test sentence",
            "we like to test our code",
            "code is fun to write",
            "writing tests is important",
            "testing ensures quality",
            "quality code is reliable",
            "this is a test sentence with more words",
            "this is what we want to test",
            "we have to make sure our code works",
            "the code must be tested properly",
            "testing is an essential part of development",
            "we like to write quality code",
            "a good developer writes tests",
            "quality and reliability are important",
            "this test should generate better results"
        ]
        
        cleaned_lines = [Preprocessor.clean(line) for line in training_lines]
        
        model = NGramLanguageModel(max_n_gram=2)
        model.train(text_lines=cleaned_lines, min_count=2)
        
        self.assertIn("this", model.ngram_probs[1][""])
        self.assertIn("test", model.ngram_probs[1][""])
        self.assertIn("this", model.ngram_probs[2])
        self.assertIn("test", model.ngram_probs[2])
        
        generated_text = model.generate(seed_text="", max_words=10, use_top_candidate=1)
        self.assertEqual(generated_text, "")
        
        generated_text = model.generate(seed_text="this is", max_words=10, use_top_candidate=1)
        self.assertTrue(generated_text.startswith("this is"))

    def test_save_and_load(self):
        """Test saving and loading the model."""
        model = NGramLanguageModel(max_n_gram=2)
        model.train(text_lines=["this is a test"], min_count=1)
        
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            model_path = Path(tmp.name)
            model.save(file_path=model_path)
            loaded_model = NGramLanguageModel.load(file_path=model_path)
            
            self.assertEqual(model.max_n_gram, loaded_model.max_n_gram)
            
            for n in range(1, model.max_n_gram + 1):
                self.assertEqual(
                    set(model.ngram_probs[n].keys()),
                    set(loaded_model.ngram_probs[n].keys())
                )
            
            seed_text = "this is"
            original_text = model.generate(seed_text=seed_text, max_words=5, use_top_candidate=1)
            loaded_text = loaded_model.generate(seed_text=seed_text, max_words=5, use_top_candidate=1)
            
            self.assertEqual(original_text, loaded_text)
            
    def test_unknown_context(self):
        """Test generating with an unknown context."""
        model = NGramLanguageModel(max_n_gram=2)
        model.train(text_lines=["this is a test"], min_count=1)
        
        generated_text = model.generate(seed_text="unknown context", max_words=10, use_top_candidate=1)
        self.assertEqual(generated_text, "unknown context")
        
    def test_backoff_functionality(self):
        """Test the backoff functionality of the language model."""
        training_lines = [
            "the cat sat on the mat",
            "the dog sat on the floor",
            "the cat chased the mouse",
            "a bird flew over the house",
            "the dog barked at the cat",
            "the mouse ran away quickly",
            "the cat meowed loudly"
        ]
        
        cleaned_lines = [Preprocessor.clean(line) for line in training_lines]
        model = NGramLanguageModel(max_n_gram=3)
        model.train(text_lines=cleaned_lines, min_count=1)
        
        generated_text = model.generate(seed_text="the cat", max_words=5, use_top_candidate=1)
        self.assertTrue(
            generated_text.startswith("the cat") and "the cat sat" in generated_text,
            f"Expected generation to start with 'the cat sat', got '{generated_text}'"
        )
        
        generated_text = model.generate(seed_text="dog sat", max_words=5, use_top_candidate=1)
        self.assertTrue(
            generated_text.startswith("dog sat") and "dog sat on" in generated_text,
            f"Expected generation to start with 'dog sat on', got '{generated_text}'"
        )

    def test_adaptive_topk(self):
        """Test the adaptive top-k functionality."""
        training_lines = [
            # Certain context - "red apple" is very predictable
            "red apple red apple red apple red apple red apple",
            "red apple red apple red apple red apple red apple",
            
            # Uncertain context - "blue" has many possible continuations
            "blue sky blue water blue paint blue light blue shirt",
            "blue car blue bird blue flower blue eyes blue book"
        ]
        
        cleaned_lines = [Preprocessor.clean(line) for line in training_lines]
        model = NGramLanguageModel(max_n_gram=2)
        model.train(text_lines=cleaned_lines, min_count=1)
        
        certain_results = set()
        for _ in range(5):
            generated = model.generate(seed_text="red", max_words=1, use_top_candidate=5)
            certain_results.add(generated)
        
        uncertain_results = set()
        for _ in range(10):
            generated = model.generate(seed_text="blue", max_words=1, use_top_candidate=5) 
            uncertain_results.add(generated)
        
        self.assertLessEqual(
            len(certain_results), 
            len(uncertain_results), 
            f"Expected uncertain context to produce more variety, but got {len(certain_results)} vs {len(uncertain_results)}"
        )


if __name__ == "__main__":
    unittest.main()