Text generation using an $n$-gram language model
=====

For this kata, we want you to create a text generation tool using an $n$-gram language model.
In other words, you will train a model that will predict which is the most probable word
to follow a given $n$-gram. The approach presented here is similar to the predictive
keyboards, such as those find in our mobile phones.

$$\text{I work in a DS project for Jus}\left\{
  \begin{aligned}
    Mundi: 0.40 \\
    AI: 0.35 \\
    Connect: 0.25 \\
  \end{aligned}
\right.$$

To do that, you will program a code for _training_ an $n$-gram language model, as well, as a code
for reading this model and generating some text.

In the following sections, we will present you with all the necessary aspects that you will need to
know about the expected code, such as the goal and rules. In the [Wiki](../../wikis/home)
of the project, you will find
the necessary background and a series of How-to in order to understand this kata.

# Goals and rules

  In this section, we present you with the goals and rules of the kata, please read them carefully.
  Depending on your level, you might have different goals.

  ## Goals
  - Create a code for training an $n$-gram language model:
    - Level 1: The size of the $n$-gram must be 2.
    - Level 2: The size of the $n$-gram must be 3, with a backup for $n$-grams of size $2$. In other
    words, if the largest $n$-gram does not exist, it should try with a smaller one 
    (i.e., a simplified backoff approach, see the project's [Wiki](../../wikis/home)).
  - Create a code for reading the language model and generate text.
    - Level 1: The code must generate the text selecting the most probable word, i.e. top $x=1$ words.
    - Level 2: The user should be able to define from how many top $x$ words, the generator.
    can select a word to generate a text. This will be defined through the field `use_top_candidate` in the `InputSchema` (see the code provided).
  - The text generation can be triggered without defining a starting text (i.e., empty string). 
  ### Optional goals

  - Let the user define the size of the $n$-gram language model to train.
  - Let the user define the maximum size of the $n$-gram to use during the generation.
  - Implement for any $n$-gram language model a simple backoff approach (see the project's [Wiki](../../wikis/home)). 
  In other words, if the largest $n$-gram does not exist, it should try with a smaller one.

  ## Rules to follow
  
  We have defined some rules that you must follow:
  - Do not change the name or path of the scripts files `train.py` and `generate.py`.
    Also, do not change the input arguments or the main functions signatures (`main_train` and `main_generate`).   
  - Use the input/output schemas defined by us (see the project's [Wiki](../../wikis/home)) and do not modify them.
  - For the training code:
    - Do not calculate the probabilities for $n$-grams occurring fewer than 6 times.
    - The input data are plain text documents located in a directory.
    - Do not enforce a particular file name extension for the trained model.
     In other words, the user is free to define the file name including the extension (e.g. .file, .jm, .data, etc.). 
  - For all the input texts (training/generation):
    - Only use the text pre-processor provided by us. In other words, do not pre-process more the input texts.
    - Tokenize them based on white-spaces only.
    - Never change the case of the input/output.
    - The new line character (\n) will mark the end of a string.
    - Do not generate $n$-grams over multiple lines. In other words the new line character
    cannot be part of an $n$-gram. 
  - For the generated text:
    - Do not print the symbols marking the beginning or ending of a text (see the project's [Wiki](../../wikis/home)).
    - Ensure that the generated text do not start or end with any kind of spaces.
    - The generation of a text must end when:
      - The end of a string has been predicted.
      - There is no following word in the model.
      - More than 50 words have been generated (do not count those input by the user).
  - Do not use external tools, such as Spacy for processing the text.
    We want to see your thinking and coding competences.
  - Do not change the minimum version of Python (3.10).
  - Do not remove any dependency found in either the `pyproject.toml` or the `requirements.txt` files.
  - You can add new entries to the .gitignore file, but do not remove entries from it.
  - You do not need to provide neither the training data nor the trained $n$-grams language model.
  - You must commit your code on GitHub or any other SCM repository you prefer (e.g. BitBucket, GitLab, etc) and send us the link. 
  - Read carefully this README, and do not take more than 2 hours to develop the code.

# What we provide

 We provide some basic code, you can modify them based on your
 needs while respecting the defined rules of the kata. This code will help you on the
 most basic tasks, such as validating the JSON file, reading a directory or creating a JSON file.

 Moreover, we provide the project's depencencies in two formats: Poetry (`pyproject.toml`) and
 `requirements.txt`. While we support both for this kata, we prefer the former one.
 Both files contain the necessary packages to install in order to run the provided code. Furthermore, do not remove any dependency found in the dependecies.

# Expected Code

 - If there is a bug, feel free to fix it.
 - Try to create your best code possible.
 - If you need to add external packages, please add them to the pyproject.toml file, and, if possible, lock it. However, do not remove those already present.
 - Be sure that your code runs.
 - No matter if you do not finish the kata, we will assess your code.

# Evaluation by Jus Mundi

 The evaluation will consist in two aspects:
 - We will use an evaluation benchmark to verify that the code works as expected.
 Thus, it is imperative that the name or path of `generate.py` and `train.py` do not change, neither the
signature of the main functions (`main_train` and `main_generate`). Furthermore, do not change the JSON schemas.
 - We will assess manually your code, quality, organization, algorithm.