"""Code for calling the generating a text."""

import os
from sys import argv
from typing import List

from TextGeneration.utils.files import json_to_schema, schema_to_json
from TextGeneration.utils.logger import generate_logger
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import InputSchema, OutputSchema
from TextGeneration.utils.language_model import NGramLanguageModel


def main_generate(file_str_path: str) -> None:
    """
    Call for generating a text.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the generation
    :return: None
    """
    try:
        generate_logger.info(f"Starting text generation: config={file_str_path}")

        try:
            input_schema = json_to_schema(
                file_str_path=file_str_path, input_schema=InputSchema
            )
        except (FileNotFoundError, ValueError) as e:
            generate_logger.error(f"Failed to load generation configuration: {e}")
            raise ValueError(f"Invalid generation configuration: {e}") from e

        if input_schema.max_n_gram < 1:
            error_msg = (
                f"Invalid max_n_gram value: {input_schema.max_n_gram}, must be >= 1"
            )
            generate_logger.error(error_msg)
            raise ValueError(error_msg)

        if input_schema.use_top_candidate < 1:
            error_msg = f"Invalid use_top_candidate value: {input_schema.use_top_candidate}, must be >= 1"
            generate_logger.error(error_msg)
            raise ValueError(error_msg)

        generate_logger.info(
            f"Generation config: model={input_schema.trained_model}, "
            f"max_n={input_schema.max_n_gram}, top_k={input_schema.use_top_candidate}, "
            f"texts={len(input_schema.texts)}, output={input_schema.output_file}"
        )

        if not input_schema.trained_model.exists():
            error_msg = f"Model file not found: {input_schema.trained_model}"
            generate_logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            model = NGramLanguageModel.load(file_path=input_schema.trained_model)
        except (ValueError, IOError, FileNotFoundError) as e:
            generate_logger.error(f"Failed to load model: {e}")
            raise

        if not any(len(contexts) > 0 for contexts in model.ngram_probs.values()):
            generate_logger.warning(
                "Model contains no probability data, generation may be limited"
            )

        if input_schema.max_n_gram < model.max_n_gram:
            generate_logger.info(
                f"Constraining model to max_n={input_schema.max_n_gram} (from {model.max_n_gram})"
            )
            model.max_n_gram = input_schema.max_n_gram

        generated_texts: List[str] = []

        for idx, input_text in enumerate(input_schema.texts):
            cleaned_text = Preprocessor.clean(text=input_text)

            generate_logger.info(
                f"Processing text {idx + 1}/{len(input_schema.texts)}: seed='{cleaned_text or 'empty'}'"
            )

            try:
                max_words = int(os.environ.get("TEXT_GEN_MAX_WORDS", 50))
                if max_words < 1:
                    generate_logger.warning(
                        f"Invalid max_words {max_words}, using default value 50"
                    )
                    max_words = 50
            except (TypeError, ValueError):
                generate_logger.warning(
                    "Invalid max_words in environment, using default value 50"
                )
                max_words = 50

            try:
                generated_text = model.generate(
                    seed_text=cleaned_text,
                    max_words=max_words,
                    use_top_candidate=input_schema.use_top_candidate,
                )
            except Exception as e:
                generate_logger.error(f"Error generating text for input {idx + 1}: {e}")
                generated_text = cleaned_text

            generated_texts.append(generated_text)

        output_schema = OutputSchema(generated_texts=generated_texts)
        try:
            input_schema.output_file.parent.mkdir(parents=True, exist_ok=True)
            schema_to_json(file_path=input_schema.output_file, schema=output_schema)
        except IOError as e:
            generate_logger.error(f"Failed to save output: {e}")
            raise IOError(f"Failed to save generation output: {e}") from e

        generate_logger.info(
            f"Generation complete: {len(generated_texts)} texts saved to {input_schema.output_file}"
        )

    except Exception as e:
        generate_logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main_generate(file_str_path=argv[1])
