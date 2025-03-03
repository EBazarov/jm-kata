"""Different utils related to files."""

from pathlib import Path
from typing import Iterator, TypeVar, Type

from pydantic import BaseModel

from TextGeneration.utils.schemas import _BaseInputSchema

_InputSchema = TypeVar("_InputSchema", bound=_BaseInputSchema)


def json_to_schema(
    file_str_path: str, input_schema: Type[_InputSchema]
) -> _InputSchema:
    """
    Read a json file and convert it into a Schema.

    :param file_str_path: A path in string format pointing towards a json file
    :param input_schema: The type of _InputSchema to use,
     bounded to any BaseInputSchema
    :return: An object of the type defined by input_schema
    :raises: ValidationError if the JSON does not follow the schema
    """
    file_path: Path = Path(file_str_path)
    file_content: str = file_path.read_text()
    return input_schema.model_validate_json(json_data=file_content, strict=True)


def schema_to_json(file_path: Path, schema: BaseModel) -> None:
    """
    Convert a schema into a JSON file.

    :param file_path: The path where the JSON file should be saved
    :param schema: Any Pydantic BaseModel to convert into a JSON
    :return: None
    """
    with file_path.open("w") as file_content:
        json_content = schema.model_dump_json(indent=2)
        file_content.write(json_content)


def read_dir(dir_path: Path) -> Iterator[str]:
    """
    Read each line of each .txt file in a directory.

    :param dir_path: The path of the directory to read
    :return: A string iterator
    """
    for file_path in dir_path.glob("*.txt"):
        with file_path.open("r") as file_content:
            while line := file_content.readline():
                yield line
