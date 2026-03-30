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

modelPaths = jobSettings.get("modelPaths") or [jobSettings["modelPath"]]
settingsFileNames = jobSettings.get("settingsFileNames") or [jobSettings["settingsFileName"]]

for modelPath in modelPaths:
    for settingsFileName in settingsFileNames:
        llm = loadModel(
        settingsFileName=settingsFileName,
        modelPath=modelPath
        )
        print(f"Using settings '{settingsFileName}'")
        for input in jobSettings["inputs"]:
            print(f"Generating {jobSettings['quantity']}x '{input}'")
            for i in range(jobSettings['quantity']):
                print(f"Generating {i + 1} of {jobSettings['quantity']}")
                parsed = runPrompt(input=input, prompt=prompt, llm=llm, outputParser=output_parser)
                output_text = parsed if isinstance(parsed, str) else (parsed.get("prompt") if parsed else None)
                if output_text:
                    prefix = '' if len(settingsFileNames) == 1 else f"{settingsFileName}-"
                    if (jobSettings['outputPath']):
                        file1 = open(f"{jobSettings['outputPath']}/{prefix}{jobSettings['templateFileName'].split('.')[0]}-{input.split()[0]}.txt", "a")
                        file1.write(output_text + "\n")
                        file1.close()
                    else:
                        print(output_text)
        del(llm)
