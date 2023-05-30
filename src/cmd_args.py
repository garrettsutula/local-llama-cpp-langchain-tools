import argparse, os
from dotenv import dotenv_values

parser = argparse.ArgumentParser(description="SD.Next", conflict_handler='resolve', epilog='For other options see UI Settings page', prog='', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))
parser._optionals = parser.add_argument_group('Other options') # pylint: disable=protected-access
group = parser.add_argument_group('cli options')

group.add_argument('-m', "--model", type=str, help="Path to model file to load.")
group.add_argument("-t", "--prompt-template", type=str, help="Prompt file name to load and run from './prompt_templates/'")
group.add_argument("-i", "--input", type=str, help="Prompt input", required=True)
group.add_argument("-q", "--quantity", type=int, default=5)
group.add_argument("-o", "--output", help="Output path, print to console if not provided as a cli arg or .env")
group.add_argument("-s", "--settings", help="Settings file to load from './settings/', llama-cpp-python defaults used if not passed")

def get_args():
    args = parser.parse_args()
    config = dotenv_values(".env")

    if (args.output):
        args.output_path = args.output
    elif (config.get("output")):
        args.output_path = config["output"]
    else:
        args.output_path = None

    if (args.model):
        args.model_path = args.model
    elif (config.get("model")):
        args.model_path = config["model"]
    else:
        raise Exception("Model path must be specified in .env or cli arg (-m, --model)")

    if (args.settings):
        args.settings = args.settings
    elif (config.get("settings")):
        args.settings = config["settings"]

    # Load prompt template from yaml file
    entries = os.listdir('prompt_templates/')
    if (not args.prompt_template in entries):
        raise Exception("Prompt template not found in ./prompt_templates, include full file name")
    
    return args