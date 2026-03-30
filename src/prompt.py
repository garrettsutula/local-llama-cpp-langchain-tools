import yaml

from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_classic.prompts import FewShotPromptTemplate, PromptTemplate

def loadFewShot(templateFileName: str):
  prompt_template = None
  with open(f"./prompt_templates/{templateFileName}", "r") as stream:
      try:
          prompt_template = yaml.safe_load(stream)
      except yaml.YAMLError as e:
          print(e)

  output_parser = None
  if prompt_template.get("response_schemas"):
      response_schemas = list(map(lambda prop: ResponseSchema(name=prop["name"], description=prop["description"]), prompt_template["response_schemas"]))
      output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)

  example_prompt = PromptTemplate(
      template=prompt_template["template"],
      input_variables=prompt_template["template_input_variables"]
  )

  prompt = FewShotPromptTemplate(
      examples=prompt_template["examples"],
      example_prompt=example_prompt,
      prefix=prompt_template["prefix"],
      suffix=prompt_template["suffix"],
      input_variables=prompt_template["input_variables"],
      output_parser=output_parser
  )

  return [ prompt, output_parser ]

def runPrompt(
        input: str,
        prompt: PromptTemplate,
        llm,
        outputParser: StructuredOutputParser
        ) -> str:
    _input = prompt.format_prompt(input=input)
    output = llm.invoke(_input.to_string())
    if outputParser:
        try:
            parsed = outputParser.parse(output)
            return parsed
        except:
            print("Parsing failed, skipping saving prompt")
            print(f"Output:\n\n{output}")
    else:
        return output
