import json
from pathlib import Path


def test_colab_chat_notebook_is_valid():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "Vnlp_scale_Chat_Demo.ipynb"
    module_paths = [
        root / "examples" / "colab_models.py",
        root / "examples" / "colab_runtime.py",
        root / "examples" / "colab_ui.py",
    ]
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    module_sources = [path.read_text(encoding="utf-8") for path in module_paths]

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"

    code_cells = [
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    combined = "\n".join(code_cells + module_sources)
    required_markers = (
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "google/gemma-3-1b-it",
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
        "BitsAndBytesConfig",
        "CUSTOM_MODEL_ID",
        "MultiModelRuntime",
        "TorchLlamaEngine.from_store",
        "gr.ChatInterface",
        "allow_tags=False",
        "transformers @ git+https://github.com/huggingface/transformers.git@main",
    )
    for marker in required_markers:
        assert marker in combined

    for path, source in zip(module_paths, module_sources, strict=True):
        compile(source, str(path), "exec")
    for index, source in enumerate(code_cells):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    models_namespace: dict = {}
    exec(compile(module_sources[0], str(module_paths[0]), "exec"), models_namespace)
    profile = models_namespace["model_profile"]
    normalize = models_namespace["normalize_token_ids"]

    assert profile("Qwen3.5 0.8B")["loader"] == "multimodal"
    assert profile("TinyLlama 1.1B Chat (Vnlp-scale compressed)")["backend"] == "vnlp"
    assert profile("Qwen3 0.6B", load_in_4bit=True)["load_in_4bit"] is True
    assert (
        profile("Qwen3 0.6B", custom_model_id="org/custom")["model_id"] == "org/custom"
    )

    class BatchEncodingLike:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    assert normalize([1, 2, 3]) == [1, 2, 3]
    assert normalize({"input_ids": [4, 5]}) == [4, 5]
    assert normalize(BatchEncodingLike([[6, 7]])) == [6, 7]
