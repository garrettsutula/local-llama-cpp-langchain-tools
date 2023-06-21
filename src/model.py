import yaml
from langchain.llms import LlamaCpp

def loadModel(settingsFileName: str, modelPath: str) -> LlamaCpp:
  settings = {}
  if (settingsFileName):
      with open(f"./settings/{settingsFileName}", "r") as stream:
          try:
              settings = yaml.safe_load(stream)
          except yaml.YAMLError as e:
              print(e)

  return LlamaCpp(
      model_path=f"{modelPath}", 
      n_gpu_layers=settings.get("n_gpu_layers"), 
      n_ctx=settings.get("n_ctx") or 512,
      max_tokens=settings.get("max_tokens") or 128,
      temperature=settings.get("temp"),
      top_k=settings.get("top_k"),
      top_p=settings.get("top_p"),
      repeat_penalty=settings.get("repeat_penalty")
  )