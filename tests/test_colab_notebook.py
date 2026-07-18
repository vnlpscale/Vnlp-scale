import ast
import json
from pathlib import Path


def _function_namespace(source: str, function_name: str) -> dict:
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(functions) == 1
    namespace: dict = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


def test_colab_chat_notebook_is_valid():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "Vnlp_scale_Chat_Demo.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"

    code_cells = [
        "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
    ]
    combined = "\n".join(code_cells)
    assert "TorchLlamaEngine.from_store" in combined
    assert "gr.ChatInterface" in combined
    assert "tokenizer.apply_chat_template" in combined
    assert "normalize_token_ids" in combined
    assert "fused_linear" in combined
    assert "triton_quant" in combined

    for index, source in enumerate(code_cells):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    normalize_source = next(source for source in code_cells if "def normalize_token_ids" in source)
    normalize = _function_namespace(normalize_source, "normalize_token_ids")["normalize_token_ids"]

    class BatchEncodingLike:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class ArrayLike:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    assert normalize([1, 2, 3]) == [1, 2, 3]
    assert normalize({"input_ids": [4, 5]}) == [4, 5]
    assert normalize(BatchEncodingLike([[6, 7]])) == [6, 7]
    assert normalize(ArrayLike([[8, 9]])) == [8, 9]
