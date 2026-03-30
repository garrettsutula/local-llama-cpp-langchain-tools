import yaml
from langchain_community.llms import LlamaCpp


def load_model(settings_file_name: str, model_path: str) -> LlamaCpp:
    """Load a LlamaCpp LLM from a YAML settings file and model path.

    Args:
        settings_file_name: File name (not full path) inside ./settings/.
        model_path: Absolute or relative path to the GGUF/GGML model binary.

    Returns:
        A configured LlamaCpp instance ready for invocation.
    """
    settings: dict = {}
    with open(f"./settings/{settings_file_name}", "r") as fh:
        try:
            settings = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            print(f"Failed to parse settings file '{settings_file_name}': {exc}")

    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=settings.get("n_gpu_layers", 0),
        n_ctx=settings.get("n_ctx", 512),
        max_tokens=settings.get("max_tokens", 128),
        temperature=settings.get("temp", 0.8),
        top_k=settings.get("top_k", 40),
        top_p=settings.get("top_p", 0.95),
        repeat_penalty=settings.get("repeat_penalty", 1.1),
        use_mlock=False,
        verbose=True,
    )
