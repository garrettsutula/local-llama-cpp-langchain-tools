from cmd_args import get_args
from model import loadModel
from prompt import loadFewShot, runPrompt

args = get_args()

prompt, output_parser = loadFewShot(templateFileName=args.prompt_template)

llm = loadModel(settingsFileName=args.settings, modelPath=args.model_path)

for i in range(args.quantity):
    print(f"Generating {i + 1} of {args.quantity}")
    parsed = runPrompt(input=args.input, prompt=prompt, llm=llm, outputParser=output_parser)
    if (parsed["prompt"]):
        if (args.output_path):
            file1 = open(f"{args.output_path}/{args.prompt_template.split('.')[0]}-{args.input.split()[0]}.txt", "a")
            file1.write(parsed["prompt"] + "\n")
            file1.close()
        else:
            print(parsed["prompt"])
