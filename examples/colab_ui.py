"""Gradio UI for the multi-model Colab chat demo."""

from __future__ import annotations

from colab_runtime import MultiModelRuntime


def launch_chat(
    runtime: MultiModelRuntime,
    *,
    system_prompt: str = "You are a concise and helpful assistant.",
):
    import threading
    import gradio as gr

    lock = threading.Lock()

    def history_messages(history):
        result = []
        for item in history or []:
            role = (
                item.get("role")
                if isinstance(item, dict)
                else getattr(item, "role", None)
            )
            content = (
                item.get("content")
                if isinstance(item, dict)
                else getattr(item, "content", None)
            )
            if role in {"user", "assistant"} and isinstance(content, str):
                result.append({"role": role, "content": content})
        return result

    def respond(message, history, max_new_tokens, temperature, sample, top_p, top_k):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages(history))
        messages.append({"role": "user", "content": (message or "").strip()})
        with lock:
            answer, stats = runtime.generate(
                messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                sample=bool(sample),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        info = stats.get("runtime", {})
        return answer + (
            f"\n\n---\n`model={runtime.model_id}` · `backend={runtime.backend}` · "
            f"`{stats['tokens_per_second']:.2f} tok/s` · "
            f"`{stats['generated_tokens']} tokens` · `dtype={info.get('dtype', 'unknown')}`"
        )

    profile = runtime.profile
    demo = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(type="messages", height=560, allow_tags=False),
        textbox=gr.Textbox(
            placeholder=f"Ask {runtime.model_id} something…", container=False
        ),
        title="Vnlp-scale Multi-Model Chat Demo",
        description=f"Current model: {runtime.model_id} | backend: {runtime.backend} | loader: {runtime.loader}",
        additional_inputs=[
            gr.Slider(1, 512, value=64, step=1, label="Maximum new tokens"),
            gr.Slider(
                0.05,
                1.5,
                value=float(profile.get("temperature", 0.7)),
                step=0.05,
                label="Temperature",
            ),
            gr.Checkbox(value=True, label="Sample"),
            gr.Slider(
                0.05,
                1.0,
                value=float(profile.get("top_p", 0.9)),
                step=0.05,
                label="Top-p",
            ),
            gr.Slider(
                1, 100, value=int(profile.get("top_k", 50)), step=1, label="Top-k"
            ),
        ],
        examples=[
            ["日本語で短く自己紹介してください。"],
            ["Explain compressed-stage matrix multiplication."],
        ],
    )
    return demo.queue(default_concurrency_limit=1)
