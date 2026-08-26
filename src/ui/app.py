"""Gradio UI for ResearchOS.

Talks to the FastAPI app over HTTP rather than importing the pipeline, so the
demo exercises the same path a real client would and the API stays the single
place where queries and clicks are logged.

    python scripts/serve_ui.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import httpx

MAX_CITATIONS = 5
REQUEST_TIMEOUT = 120.0

EXAMPLES = [
    "0-dimensional biomaterials show inductive properties.",
    "Vitamin D supplementation reduces the risk of respiratory tract infection.",
    "Aspirin lowers the risk of colorectal cancer.",
    "Sleep deprivation impairs immune function.",
]


def _post(api_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"{api_url.rstrip('/')}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def build_demo(api_url: str) -> gr.Blocks:
    """Construct the Gradio app bound to a running API."""

    def _blank(message: str) -> list[Any]:
        return [gr.update(value=message), None, [], ""] + [
            gr.update(visible=False) for _ in range(MAX_CITATIONS * 2)
        ]

    def ask(question: str) -> list[Any]:
        question = (question or "").strip()
        if len(question) < 3:
            return _blank("Enter a claim or question first.")

        try:
            result = _post(api_url, "/ask", {"query": question, "top_k": MAX_CITATIONS})
        except Exception as exc:  # noqa: BLE001 - show transport errors in the UI
            return _blank(f"**API error:** {exc}")

        citations = result.get("citations", [])
        timing = result.get("timing_ms", {})
        footer = (
            f"backend `{result.get('backend')}` | "
            f"retrieve+rerank {timing.get('retrieval_and_rerank')} ms | "
            f"generate {timing.get('generation')} ms"
        )
        if not result.get("generation_configured"):
            footer += " | `ANTHROPIC_API_KEY` unset - passages only"

        updates: list[Any] = [
            gr.update(value=result.get("answer") or "_No answer generated._"),
            result.get("query_id"),
            [c["doc_id"] for c in citations],
            footer,
        ]
        for i in range(MAX_CITATIONS):
            if i < len(citations):
                c = citations[i]
                label = f"[{c['doc_id']}] {c['title'][:90]} (score {c['score']:.2f})"
                updates.append(gr.update(visible=True, label=label))
                updates.append(gr.update(value=c["text"]))
            else:
                updates.append(gr.update(visible=False))
                updates.append(gr.update(value=""))
        return updates

    def make_click_handler(index: int):
        def handler(query_id: int | None, doc_ids: list[str]) -> str:
            if query_id is None or not doc_ids or index >= len(doc_ids):
                return "Ask something first."
            doc_id = doc_ids[index]
            try:
                stats = _post(
                    api_url, "/feedback/click", {"query_id": int(query_id), "doc_id": doc_id}
                )
            except Exception as exc:  # noqa: BLE001
                return f"Click not logged: {exc}"
            return f"Logged click on [{doc_id}] - {stats['clicks']} clicks recorded so far."

        return handler

    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    with gr.Blocks(title="ResearchOS") as demo:
        gr.Markdown(
            "# ResearchOS\n"
            "Scientific claim verification over SciFact. BM25 + dense retrieval, fused with "
            "RRF, reranked by a cross-encoder fine-tuned on **hard negatives**, answered by "
            "Claude with span citations."
        )

        with gr.Row():
            question_box = gr.Textbox(
                label="Claim or question",
                placeholder="e.g. Vitamin D supplementation reduces respiratory infection risk.",
                scale=4,
            )
            submit = gr.Button("Search", variant="primary", scale=1)

        gr.Examples(examples=EXAMPLES, inputs=question_box)

        answer_box = gr.Markdown(label="Answer")
        footer_box = gr.Markdown()
        query_id_state = gr.State(None)
        doc_ids_state = gr.State([])
        click_status = gr.Markdown()

        gr.Markdown("### Retrieved passages")
        accordions: list[gr.Accordion] = []
        bodies: list[gr.Markdown] = []
        for i in range(MAX_CITATIONS):
            with gr.Accordion(label=f"Passage {i + 1}", open=(i == 0), visible=False) as acc:
                body = gr.Markdown()
                mark = gr.Button("Mark as useful", size="sm")
                mark.click(
                    make_click_handler(i),
                    inputs=[query_id_state, doc_ids_state],
                    outputs=click_status,
                )
            accordions.append(acc)
            bodies.append(body)

        outputs: list[Any] = [answer_box, query_id_state, doc_ids_state, footer_box]
        for acc, body in zip(accordions, bodies):
            outputs.extend([acc, body])

        submit.click(ask, inputs=question_box, outputs=outputs)
        question_box.submit(ask, inputs=question_box, outputs=outputs)

    return demo
