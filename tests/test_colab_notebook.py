import json
from pathlib import Path


def test_colab_chat_notebook_is_valid():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "Vnlp_scale_Chat_Demo.ipynb"
    )
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"

    code_cells = [
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    combined = "\n".join(code_cells)
    assert "TorchLlamaEngine.from_store" in combined
    assert "gr.ChatInterface" in combined
    assert "tokenizer.apply_chat_template" in combined
    assert "fused_linear" in combined
    assert "triton_quant" in combined

    for index, source in enumerate(code_cells):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")
