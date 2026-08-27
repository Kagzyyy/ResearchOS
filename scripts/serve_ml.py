"""Run the ResearchOS-ML backend.

    python scripts/serve_ml.py --config configs/ml.yaml

The Next.js frontend in web/ talks to this. Start both:

    python scripts/serve_ml.py            # terminal 1, port 8100
    cd web && npm run dev                 # terminal 2, port 3000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ["RESEARCHOS_ML_CONFIG"] = args.config

    if not os.environ.get("GROQ_API_KEY"):
        print("[serve] GROQ_API_KEY not set - retrieval works, answers will not be generated.")
        print("[serve] Free key: https://console.groq.com/keys")

    import uvicorn

    uvicorn.run(
        "src.api.ml_app:app",
        host=args.host or cfg["server"]["host"],
        port=args.port or cfg["server"]["api_port"],
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
