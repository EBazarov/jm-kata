"""Code for calling the training of the model."""

import os
from sys import argv
from typing import List

from TextGeneration.utils.files import json_to_schema, read_dir
from TextGeneration.utils.logger import train_logger
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import TrainingInputSchema
from TextGeneration.utils.language_model import NGramLanguageModel


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    try:
        train_logger.info(f"Starting n-gram model training: config={file_str_path}")

        try:
            training_schema = json_to_schema(
                file_str_path=file_str_path, input_schema=TrainingInputSchema
            )
        except (FileNotFoundError, ValueError) as e:
            train_logger.error(f"Failed to load training configuration: {e}")
            raise ValueError(f"Invalid training configuration: {e}") from e

        if training_schema.max_n_gram < 1:
            error_msg = (
                f"Invalid max_n_gram value: {training_schema.max_n_gram}, must be >= 1"
            )
            train_logger.error(error_msg)
            raise ValueError(error_msg)

        train_logger.info(
            f"Training config: max_n={training_schema.max_n_gram}, "
            f"input={training_schema.input_folder}, "
            f"output={training_schema.trained_model}"
        )

        if not training_schema.input_folder.exists():
            error_msg = f"Input folder does not exist: {training_schema.input_folder}"
            train_logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        cleaned_lines: List[str] = []
        try:
            for training_line in read_dir(dir_path=training_schema.input_folder):
                cleaned_line = Preprocessor.clean(text=training_line)
                if cleaned_line.strip():
                    cleaned_lines.append(cleaned_line)
        except Exception as e:
            train_logger.error(f"Error reading training data: {e}")
            raise IOError(f"Failed to read training data: {e}") from e

        if not cleaned_lines:
            warning_msg = "No valid training data found, model will be empty"
            train_logger.warning(warning_msg)
        else:
            train_logger.info(
                f"Preprocessing complete: {len(cleaned_lines)} lines collected"
            )

        model = NGramLanguageModel(max_n_gram=training_schema.max_n_gram)

        try:
            min_count = int(os.environ.get("TEXT_GEN_DEFAULT_MIN_COUNT", 6))
            if min_count < 1:
                train_logger.warning(
                    f"Invalid min_count {min_count}, using default value 6"
                )
                min_count = 6
        except (TypeError, ValueError):
            train_logger.warning(
                "Invalid min_count in environment, using default value 6"
            )
            min_count = 6

        model.train(text_lines=cleaned_lines, min_count=min_count)

        try:
            model.save(file_path=training_schema.trained_model)
        except IOError as e:
            train_logger.error(f"Failed to save model: {e}")
            raise

        train_logger.info("Training completed successfully")

    except Exception as e:
        train_logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main_train(file_str_path=argv[1])
