import yaml

from cmd_args import get_args
from model import loadModel
from prompt import loadFewShot, runPrompt

args = get_args()

jobSettings = None
with open(f"./jobs/{args.name}", "r") as stream:
    try:
        jobSettings = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)

prompt, output_parser = loadFewShot(templateFileName=jobSettings["templateFileName"])

llm = loadModel(
    settingsFileName=jobSettings["settingsFileName"],
    modelPath=jobSettings["modelPath"]
    )

for input in jobSettings["inputs"]:
    for i in range(jobSettings['quantity']):
        print(f"Generating {i + 1} of {jobSettings['quantity']}")
        parsed = runPrompt(input=input, prompt=prompt, llm=llm, outputParser=output_parser)
        if (parsed["prompt"]):
            if (jobSettings['outputPath']):
                file1 = open(f"{jobSettings['outputPath']}/{jobSettings['templateFileName'].split('.')[0]}-{input.split()[0]}.txt", "a")
                file1.write(parsed["prompt"] + "\n")
                file1.close()
            else:
                print(parsed["prompt"])
