import os
from dotenv import dotenv_values
import yaml
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

from langchain.llms import LlamaCpp
from langchain import FewShotPromptTemplate, PromptTemplate

import cmd_args

args, _ = cmd_args.parser.parse_known_args()
config = dotenv_values(".env")
model = ""
output_path = "./output/"
settings_path = None

if (args.output):
    output_path = args.output
elif (config.get("output")):
    output_path = config["output"]
else:
    output_path = None

if (args.model):
    model = args.model
elif (config.get("model")):
    model = config["model"]
else:
    raise Exception("Model path must be specified in .env or cli arg (-m, --model)")

if (args.settings):
    settings_path = args.settings
elif (config.get("settings")):
    settings_path = config["settings"]

# Load prompt template from yaml file
entries = os.listdir('prompt_templates/')
if (not args.prompt_template in entries):
    raise Exception("Prompt template not found in ./prompt_templates, include full file name")


# Load and prepare prompt template

prompt_template = None
with open(f"./prompt_templates/{args.prompt_template}", "r") as stream:
    try:
        prompt_template = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)

response_schemas = list(map(lambda prop: ResponseSchema(name=prop["name"], description=prop["description"]), prompt_template["response_schemas"]))
output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)

format_instructions = output_parser.get_format_instructions()
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

# Load and prepare model

settings = {}
if (settings_path):
    with open(f"./settings/{args.settings}", "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

llm = LlamaCpp(
    model_path=f"{model}", 
    n_gpu_layers=settings.get("n_gpu_layers"), 
    n_ctx=settings.get("n_ctx") or 512,
    max_tokens=settings.get("max_tokens") or 128,
    temperature=settings.get("temp"),
    top_k=settings.get("top_k"),
    top_p=settings.get("top_p"),
    repeat_penalty=settings.get("repeat_penalty")
)

_input = prompt.format_prompt(input=args.input)
for i in range(args.quantity):
    print(f"Generating {i + 1} of {args.quantity}")
    output = llm(_input.to_string())
    try:
        parsed = output_parser.parse(output)
        if (output_path):
            file1 = open(f"{output_path}/{args.prompt_template.split('.')[0]}-{args.input.split()[0]}.txt", "a")
            file1.write(parsed["prompt"] + "\n")
            file1.close()
        else:
            print(parsed["prompt"])
    except:
        # no-op
        print("Parsing failed, skipping saving prompt")
        print(f"Output:\n{output}")

