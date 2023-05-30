import yaml

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain import FewShotPromptTemplate, PromptTemplate

from cmd_args import get_args

args = get_args()

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
if (args.settings_path):
    with open(f"./settings/{args.settings}", "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

llm = LlamaCpp(
    model_path=f"{args.model_path}", 
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
        if (args.output_path):
            file1 = open(f"{args.output_path}/{args.prompt_template.split('.')[0]}-{args.input.split()[0]}.txt", "a")
            file1.write(parsed["prompt"] + "\n")
            file1.close()
        else:
            print(parsed["prompt"])
    except:
        # no-op
        print("Parsing failed, skipping saving prompt")
        print(f"Output:\n{output}")

