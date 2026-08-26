"""Run the FastAPI app.

    python scripts/serve_api.py --config configs/app.yaml
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
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # The app reads its config from the environment so uvicorn's import string works.
    os.environ["RESEARCHOS_CONFIG"] = args.config

    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=args.host or cfg["server"]["host"],
        port=args.port or cfg["server"]["api_port"],
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
