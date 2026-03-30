from __future__ import annotations

from typing import Any

import yaml
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model


def _build_output_parser(
    response_schemas: list[dict[str, str]],
) -> JsonOutputParser:
    """Build a Pydantic-backed JsonOutputParser from response schema dicts.

    Each schema dict must have ``name`` and ``description`` keys, mirroring
    the legacy ResponseSchema format so existing YAML templates are unchanged.

    Args:
        response_schemas: List of ``{"name": ..., "description": ...}`` dicts
            taken directly from the template YAML.

    Returns:
        A JsonOutputParser whose format instructions will be injected into
        the prompt prefix.
    """
    field_definitions: dict[str, Any] = {
        schema["name"]: (str, Field(description=schema["description"]))
        for schema in response_schemas
    }
    pydantic_model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        "DynamicResponseModel", **field_definitions
    )
    return JsonOutputParser(pydantic_object=pydantic_model)


def load_few_shot(template_file_name: str) -> tuple[FewShotPromptTemplate, Runnable]:
    """Load a few-shot prompt template and its output parser from a YAML file.

    The YAML schema is unchanged from the original project.  If the template
    defines ``response_schemas``, a Pydantic-backed JsonOutputParser is built
    and its format instructions are appended to the prompt prefix.  Otherwise
    a plain StrOutputParser is used.

    Args:
        template_file_name: File name (not full path) inside ./prompt_templates/.

    Returns:
        A ``(prompt, output_parser)`` tuple.  The output parser is either a
        :class:`JsonOutputParser` or a :class:`StrOutputParser`.
    """
    template_config: dict = {}
    with open(f"./prompt_templates/{template_file_name}", "r") as fh:
        try:
            template_config = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            print(f"Failed to parse template file '{template_file_name}': {exc}")

    # --- output parser -------------------------------------------------
    output_parser: Runnable
    format_instructions: str | None = None

    if template_config.get("response_schemas"):
        output_parser = _build_output_parser(template_config["response_schemas"])
        format_instructions = output_parser.get_format_instructions()
    else:
        output_parser = StrOutputParser()

    # --- prompt --------------------------------------------------------
    example_prompt = PromptTemplate(
        template=template_config["template"],
        input_variables=template_config["template_input_variables"],
    )

    prefix = template_config["prefix"]
    if format_instructions:
        prefix = f"{prefix}\n{format_instructions}"

    prompt = FewShotPromptTemplate(
        examples=template_config["examples"],
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=template_config["suffix"],
        input_variables=template_config["input_variables"],
    )

    return prompt, output_parser


def build_chain(
    template_file_name: str,
    llm: Any,
) -> Runnable:
    """Build a complete LCEL inference chain for a given template and LLM.

    The chain is: ``prompt | llm | output_parser``

    The prompt is a :class:`FewShotPromptTemplate`.  LlamaCpp produces a raw
    string, so the LLM output passes directly into the output parser without
    any intermediate conversion step.

    Args:
        template_file_name: File name (not full path) inside ./prompt_templates/.
        llm: Any LangChain-compatible LLM (typically a :class:`LlamaCpp` instance).

    Returns:
        A :class:`Runnable` that accepts ``{"input": str}`` and returns either
        a plain string (StrOutputParser) or a parsed dict (JsonOutputParser).
    """
    prompt, output_parser = load_few_shot(template_file_name)
    chain: Runnable = prompt | llm | output_parser
    return chain
