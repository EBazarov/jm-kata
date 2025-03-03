"""Cleaning of a text."""

import re


class Preprocessor:
    """Class that implements the cleaning of a text."""

    __new_lines = re.compile("\r\n?")
    __spaces = re.compile(r"\s+")
    __start_spaces = re.compile(r"^ ")
    __end_spaces = re.compile(r" $")
    __underscore = re.compile(r"_")
    __quotes = re.compile(r"[“”\"]")
    __quote = re.compile(r"’")
    __dash = re.compile(r"—|--")

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Clean the text and standardize it.

        :param text: Text to clean
        :return: Cleaned text
        """
        text = cls.__new_lines.sub("\n", string=text)
        text = cls.__underscore.sub(" ", text)
        text = cls.__quotes.sub(" ", text)
        text = cls.__quote.sub("'", text)
        text = cls.__dash.sub(" - ", text)
        text = cls.__spaces.sub(repl=" ", string=text)
        text = cls.__start_spaces.sub(repl="", string=text)
        text = cls.__end_spaces.sub(repl="", string=text)
        return text
