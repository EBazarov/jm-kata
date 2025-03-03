"""Core functionality for n-gram language model training and generation."""

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

from TextGeneration.utils.logger import model_logger


class NGramLanguageModel:
    """
    Implementation of an n-gram language model with backoff capabilities and adaptive top-k selection.

    Features:
    - Supports n-grams of arbitrary size
    - Implements backoff to smaller n-grams when larger ones are not available
    - Uses adaptive top-k selection that dynamically adjusts randomness based on prediction certainty
    - Handles model serialization and deserialization
    """

    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, max_n_gram: int = 2):
        """
        Initialize the language model with specified maximum n-gram size.

        :param max_n_gram: Maximum size of n-grams to use
        """
        self.max_n_gram = max_n_gram
        self.ngram_counts: Dict[int, Dict[str, Counter]] = {}
        self.ngram_probs: Dict[int, Dict[str, Dict[str, float]]] = {}

        for n in range(1, max_n_gram + 1):
            self.ngram_counts[n] = defaultdict(Counter)
            self.ngram_probs[n] = defaultdict(dict)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on whitespace.

        :param text: Input text
        :return: List of tokens
        """
        return text.split()

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, str]]:
        """
        Generate n-grams from a list of tokens.

        :param tokens: List of tokens
        :param n: Size of n-gram
        :return: List of (context, next_word) tuples
        """
        padded_tokens = [self.BOS] * (n - 1) + tokens + [self.EOS]
        ngrams = []

        for i in range(len(padded_tokens) - n + 1):
            context = " ".join(padded_tokens[i : i + n - 1])
            next_word = padded_tokens[i + n - 1]
            ngrams.append((context, next_word))

        return ngrams

    def train(self, text_lines: List[str], min_count: int = 6) -> None:
        """
        Train the language model on a corpus.

        :param text_lines: List of text lines
        :param min_count: Minimum count for n-grams to include in probability calculations
        """
        model_logger.info(
            f"Training n-gram model: max_n={self.max_n_gram}, min_count={min_count}, corpus_size={len(text_lines)} lines"
        )

        total_tokens = 0
        skipped_lines = 0

        for line_idx, line in enumerate(text_lines):
            if not line.strip():
                skipped_lines += 1
                continue

            tokens = self._tokenize(line.strip())
            total_tokens += len(tokens)

            if len(tokens) < 1:
                skipped_lines += 1
                continue

            for n in range(1, self.max_n_gram + 1):
                for context, next_word in self._get_ngrams(tokens, n):
                    self.ngram_counts[n][context][next_word] += 1

            progress_interval = int(os.environ.get("TEXT_GEN_PROGRESS_INTERVAL", 10000))
            if (line_idx + 1) % progress_interval == 0:
                model_logger.info(
                    f"Progress: processed {line_idx + 1}/{len(text_lines)} lines ({(line_idx + 1) / len(text_lines):.1%})"
                )

        model_logger.info(
            f"Corpus stats: {total_tokens} tokens, {skipped_lines} skipped lines"
        )

        model_sizes = []

        for n in range(1, self.max_n_gram + 1):
            contexts_count = 0
            total_entries = 0
            filtered_entries = 0

            for context, next_word_counts in self.ngram_counts[n].items():
                valid_next_words = {
                    word: count
                    for word, count in next_word_counts.items()
                    if count >= min_count
                }

                total_entries += len(next_word_counts)
                filtered_entries += len(next_word_counts) - len(valid_next_words)

                if valid_next_words:
                    contexts_count += 1
                    total_count = sum(valid_next_words.values())

                    for next_word, count in valid_next_words.items():
                        self.ngram_probs[n][context][next_word] = count / total_count

            model_sizes.append(
                f"{n}-gram: {contexts_count} contexts ({filtered_entries}/{total_entries} entries filtered)"
            )

        model_logger.info(f"Model size summary: {'; '.join(model_sizes)}")
        model_logger.info("Training complete")

    def generate(
        self, seed_text: str = "", max_words: int = 50, use_top_candidate: int = 1
    ) -> str:
        """
        Generate text using the trained model.

        :param seed_text: Optional starting text
        :param max_words: Maximum number of words to generate
        :param use_top_candidate: Number of top candidates to consider
        :return: Generated text
        """

        has_seed = bool(seed_text.strip())
        seed_summary = f"'{seed_text}'" if has_seed else "empty"
        model_logger.info(
            f"Generating text: seed={seed_summary}, max_words={max_words}, top_k={use_top_candidate}"
        )

        if not self.ngram_probs:
            model_logger.warning("Generation failed: model has no probability data")
            return ""

        generated_words = []

        context_words = []
        if seed_text:
            seed_tokens = self._tokenize(seed_text)
            generated_words.extend(seed_tokens)

            context_size = min(self.max_n_gram - 1, len(seed_tokens))
            context_words = seed_tokens[-context_size:] if context_size > 0 else []

        words_generated = 0
        stop_reason = None

        while words_generated < max_words:
            next_word = self._predict_next_word(context_words, use_top_candidate)

            if next_word is None:
                stop_reason = "no prediction available"
                break
            elif next_word == self.EOS:
                stop_reason = "end of sentence"
                break

            generated_words.append(next_word)
            words_generated += 1

            context_words = (
                context_words[-(self.max_n_gram - 2) :]
                if len(context_words) >= self.max_n_gram - 1
                else context_words[:]
            )
            context_words.append(next_word)

        if not generated_words:
            model_logger.info("Generation complete: no words generated")
            return ""

        result = " ".join(generated_words)
        stop_reason = stop_reason or "max words reached"
        model_logger.info(
            f"Generation complete: {words_generated} words, stopped by {stop_reason}"
        )

        return result

    def _predict_next_word(
        self, context_words: List[str], use_top_candidate: int = 1
    ) -> Optional[str]:
        """
        Predict the next word using backoff strategy and adaptive top-k selection.

        The adaptive top-k selection dynamically adjusts the number of candidate words
        based on the entropy/uncertainty of the prediction:
        - For certain predictions (low entropy), fewer candidates are considered
        - For uncertain predictions (high entropy), more candidates are considered

        This creates more interesting and varied text while maintaining predictability
        where appropriate.

        :param context_words: Words providing context
        :param use_top_candidate: Maximum number of top candidates to consider
        :return: Predicted next word or None if no prediction possible
        """

        for n in range(self.max_n_gram, 0, -1):
            context_size = n - 1
            if context_size > len(context_words):
                continue

            context = self._get_context_string(context_words, context_size)

            if context not in self.ngram_probs[n]:
                continue

            candidates = list(self.ngram_probs[n][context].items())
            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)

            dynamic_k = self._calculate_dynamic_k(candidates, use_top_candidate)
            top_k = min(dynamic_k, len(candidates))
            if top_k == 0:
                continue

            return self._select_word_from_candidates(candidates, top_k)

        return None

    def _get_context_string(self, context_words: List[str], context_size: int) -> str:
        """Get context string from context words."""
        if context_size == 0:
            return self.BOS
        return " ".join(context_words[-context_size:])

    def _calculate_dynamic_k(
        self, candidates: List[Tuple[str, float]], use_top_candidate: int
    ) -> int:
        """Calculate dynamic k value based on entropy of distribution."""
        total_candidates = len(candidates)

        if total_candidates <= 1:
            return 1

        distribution = [prob for _, prob in candidates]
        entropy = -sum(p * math.log2(p) for p in distribution) / math.log2(
            total_candidates
        )

        return max(1, min(int(use_top_candidate * (0.5 + entropy)), total_candidates))

    def _select_word_from_candidates(
        self, candidates: List[Tuple[str, float]], top_k: int
    ) -> str:
        """Select a word from the top k candidates."""
        if top_k == 1:
            return candidates[0][0]
        else:
            top_candidates = candidates[:top_k]
            words, probs = zip(*top_candidates)
            return random.choices(words, weights=probs, k=1)[0]

    def save(self, file_path: Path) -> None:
        """
        Save the trained model to a file.

        :param file_path: Path where to save the model
        :raises IOError: If the file cannot be written
        """

        sizes = []
        for n in range(1, self.max_n_gram + 1):
            contexts_count = len(self.ngram_probs[n])
            total_entries = sum(
                len(next_words) for next_words in self.ngram_probs[n].values()
            )
            sizes.append(f"{n}-gram:{contexts_count}c/{total_entries}e")

        total_contexts = sum(len(contexts) for contexts in self.ngram_probs.values())
        model_logger.info(
            f"Saving model to {file_path}: {total_contexts} contexts [{', '.join(sizes)}]"
        )

        model_data = {
            "max_n_gram": self.max_n_gram,
            "ngram_probs": {
                str(n): {
                    context: dict(next_words)
                    for context, next_words in contexts.items()
                }
                for n, contexts in self.ngram_probs.items()
            },
        }

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            model_logger.error(f"Cannot create directory for model: {e}")
            raise IOError(f"Cannot create directory for model file: {e}") from e

        try:
            with open(file_path, "w") as f:
                json.dump(model_data, f)
        except (IOError, PermissionError) as e:
            model_logger.error(f"Failed to save model: {e}")
            raise IOError(f"Failed to save model to {file_path}: {e}") from e

    @classmethod
    def load(cls, file_path: Path) -> "NGramLanguageModel":
        """
        Load a trained model from a file.

        :param file_path: Path to the saved model
        :return: Loaded NGramLanguageModel
        :raises FileNotFoundError: If the model file does not exist
        :raises ValueError: If the model file is corrupted or invalid
        """
        model_logger.info(f"Loading model from {file_path}")

        if not file_path.exists():
            error_msg = f"Model file not found: {file_path}"
            model_logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(file_path, "r") as f:
                model_data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid model file format: {e}"
            model_logger.error(error_msg)
            raise ValueError(error_msg) from e
        except IOError as e:
            error_msg = f"Error reading model file: {e}"
            model_logger.error(error_msg)
            raise IOError(error_msg) from e

        if not isinstance(model_data, dict):
            error_msg = "Invalid model format: root element is not a dictionary"
            model_logger.error(error_msg)
            raise ValueError(error_msg)

        if "max_n_gram" not in model_data:
            model_logger.warning("Model file missing max_n_gram field, defaulting to 2")

        max_n_gram = model_data.get("max_n_gram", 2)
        model = cls(max_n_gram=max_n_gram)

        sizes = []
        total_contexts = 0

        if "ngram_probs" not in model_data:
            model_logger.warning("Model file contains no n-gram probabilities")

        try:
            for n_str, contexts in model_data.get("ngram_probs", {}).items():
                try:
                    n = int(n_str)
                except ValueError:
                    model_logger.warning(f"Skipping invalid n-gram size: {n_str}")
                    continue

                if n < 1 or n > max_n_gram:
                    model_logger.warning(f"Skipping out-of-range n-gram size: {n}")
                    continue

                context_count = len(contexts)
                total_contexts += context_count
                total_entries = sum(len(next_words) for next_words in contexts.values())

                sizes.append(f"{n}-gram:{context_count}c/{total_entries}e")

                for context, next_words in contexts.items():
                    model.ngram_probs[n][context] = next_words
        except (TypeError, AttributeError) as e:
            error_msg = f"Invalid model structure: {e}"
            model_logger.error(error_msg)
            raise ValueError(error_msg) from e

        model_logger.info(
            f"Model loaded: max_n={max_n_gram}, {total_contexts} contexts [{', '.join(sizes)}]"
        )
        return model
