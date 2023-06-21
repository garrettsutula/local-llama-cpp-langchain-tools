# Local Llama.cpp LangChain Tools (L3T)

## Getting Started

1. Clone this repository
2. Create a venv `python -m venv venv` then activate it `source venv/bin/activate`
3. Install requirements `pip install -r requirements.txt` 
> Optional, do this for much faster inference: [Follow the instructions to install GPU-accelerated version of llama-cpp-python](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast) 


## CLI Arguments
Example command: `python ./src/instruct.py -t example.yaml -i "who won the super bowl in 1976?"`

| Argument               |Required| Description                                                    |
|------------------------|-|----------------------------------------------------------------|
| `-m`, `--model`                   |✅| Path to model file to load.                                     |
| `-t`, `--prompt-template`         |✅| Prompt file name to load and run from `./prompt_templates`.      |
| `-i`, `--input`                   |✅| Prompt input                                      |
| `-q`, `--quantity`                ||Quantity of generations to run and output                      |
| `-o`, `--output`                  ||Output path. If not provided as a CLI argument or in `.env`, the result will be printed to the console. |
| `-s`, `--settings`                ||Settings path. If not provided the [defaults](https://github.com/abetlen/llama-cpp-python/blob/80066f0b802f0019395466ac090c10dcd78c97bb/llama_cpp/llama.py#L458) will be passed

## .env File

Copy and rename `.env.example` to `.env` and set the values for any of the parameters below to provide default values for model, output and generation:

| Parameter        | Description                                              |
|------------------|----------------------------------------------------------|
| `model`          | Path to the model to load.                               |
| `output`         | Path to the output directory.                            |
| `settings`       | Path to the generation settings.                         |

> **Note**: Refer to [This guide](https://txt.cohere.com/llm-parameters-best-outputs-language-ai/) for more information on how generation paremeters work.

## Prompt Templates

Templates can be defined in `./prompt_templates/**/*`, a json schema is provided as documentation and for schema validation in vscode with the [YAML](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) extension.

## Generation Settings

Generation settings can be defined in `./settings/**/*`, a json schema is provided as documentation and for schema validation in vscode with the [YAML](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) extension.
