"""Batch inference entry point.

Usage:
    python src/job.py -n example.yaml

The job YAML (from ./jobs/) drives everything: which model(s) and settings
file(s) to use, which prompt template to load, which inputs to run, how many
outputs to generate per input, and where to write the results.
"""

from __future__ import annotations

import os
import sys

import yaml
from dotenv import load_dotenv

# Allow ``python src/job.py`` invocation from the project root without
# needing an explicit PYTHONPATH by inserting src/ onto sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cmd_args import get_args
from model import load_model
from prompt import build_chain


def _output_file_name(
    settings_file_name: str,
    template_file_name: str,
    input_text: str,
    multiple_settings: bool,
) -> str:
    """Derive an output file name from job parameters.

    The naming convention is unchanged from the original implementation:
    ``<settings>-<template>-<first_word_of_input>.txt`` when multiple settings
    files are in play; ``<template>-<first_word_of_input>.txt`` when there is
    only one settings file.
    """
    template_stem = template_file_name.split(".")[0]
    input_slug = input_text.split()[0]
    settings_stem = settings_file_name.split(".")[0]
    if multiple_settings:
        return f"{settings_stem}-{template_stem}-{input_slug}.txt"
    return f"{template_stem}-{input_slug}.txt"


def _extract_text(result: object) -> str | None:
    """Pull a plain string out of a chain result regardless of parser type.

    - ``StrOutputParser`` returns a raw string.
    - ``JsonOutputParser`` returns a dict; we look for a ``"prompt"`` key to
      match the legacy behaviour, then fall back to the full dict as a string.
    """
    if isinstance(result, str):
        return result.strip() or None
    if isinstance(result, dict):
        return result.get("prompt") or str(result) or None
    return None


def run_job(job_file_name: str) -> None:
    """Execute a batch inference job defined by a YAML config file.

    Args:
        job_file_name: File name (not full path) inside ./jobs/.
    """
    with open(f"./jobs/{job_file_name}", "r") as fh:
        try:
            job: dict = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            print(f"Failed to parse job file '{job_file_name}': {exc}")
            return

    # Support both singular and plural forms for model / settings keys,
    # falling back to environment variables when neither is present.
    model_paths: list[str] = job.get("modelPaths") or [
        job.get("modelPath") or os.getenv("MODEL", "")
    ]
    settings_file_names: list[str] = job.get("settingsFileNames") or [
        job.get("settingsFileName") or os.getenv("SETTINGS", "")
    ]
    template_file_name: str = job["templateFileName"]
    output_path: str | None = job.get("outputPath") or os.getenv("OUTPUT_PATH")
    inputs: list[str] = job.get("inputs", [])
    quantity: int = job.get("quantity", 1)

    multiple_settings = len(settings_file_names) > 1

    for model_path in model_paths:
        for settings_file_name in settings_file_names:
            print(
                f"Loading model: {model_path}  |  settings: {settings_file_name}"
            )
            llm = load_model(settings_file_name, model_path)

            # Build the LCEL chain once per (model, settings) combination.
            # prompt | llm | output_parser
            chain = build_chain(template_file_name, llm)

            for input_text in inputs:
                print(
                    f"Generating {quantity}x output(s) for input: '{input_text}'"
                )

                # Use chain.batch() to run all N repetitions in one call.
                # LlamaCpp is a local, single-threaded model so batch simply
                # iterates sequentially, but the interface is forward-compatible
                # with parallel backends and makes the intent clear.
                batch_inputs = [{"input": input_text}] * quantity
                results = chain.batch(batch_inputs)

                output_file: str | None = None
                if output_path:
                    output_file = _output_file_name(
                        settings_file_name,
                        template_file_name,
                        input_text,
                        multiple_settings,
                    )
                    full_path = os.path.join(output_path, output_file)

                for idx, result in enumerate(results, start=1):
                    print(f"  Result {idx} of {quantity}")
                    text = _extract_text(result)
                    if text is None:
                        print(f"  [skipped — could not extract text from result: {result!r}]")
                        continue
                    if output_path and output_file:
                        with open(full_path, "a") as fh:
                            fh.write(text + "\n")
                    else:
                        print(text)

            # Explicitly release the model so memory is freed before loading
            # the next (model, settings) combination.
            del llm


if __name__ == "__main__":
    load_dotenv()
    args = get_args()
    run_job(args.name)
