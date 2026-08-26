"""Run the Gradio UI. The API must already be running.

    python scripts/serve_api.py --config configs/app.yaml     (terminal 1)
    python scripts/serve_ui.py  --config configs/app.yaml     (terminal 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.ui.app import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--api", default=None, help="Base URL of the running FastAPI app")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = cfg["server"]["host"]
    api_url = args.api or f"http://{host}:{cfg['server']['api_port']}"

    print(f"[ui] talking to API at {api_url}")
    demo = build_demo(api_url)
    demo.launch(
        server_name=host,
        server_port=args.port or cfg["server"]["ui_port"],
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
