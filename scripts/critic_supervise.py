"""Stage 2: build reference critiques for the trajectories (Algorithm 2).

    python scripts/critic_supervise.py --limit 300 -k 3

Reference verdicts come from HotpotQA gold answers, so this script only spends
API budget where the teacher is actually needed: the location/reason/fix
fields of trajectories that went wrong. CORRECT trajectories are labelled for
free, because Figure 7(b) fixes their form to location="none", fix="keep".

Budget-aware by construction: Groq's free tier allows 1,000 requests/day, so
the script tracks spend on disk, stops cleanly when the quota is gone, and
resumes where it left off on the next run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import src.config  # noqa: F401,E402  - loads .env

from src.critic.supervise import (  # noqa: E402
    Budget, BudgetExhausted, DEFAULT_TEACHER, SupervisedCritique, Teacher,
    diagnose_trajectory, read_supervision, supervise_trajectory,
    supervision_summary, write_supervision,
)
from src.critic.trajectory import read_trajectories  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", default="data/critic/trajectories_train.jsonl")
    ap.add_argument("--out", default="data/critic/supervision_train.jsonl")
    ap.add_argument("--budget-file", default="data/critic/.groq_budget.json")
    ap.add_argument("--limit", type=int, default=None, help="max trajectories to label")
    ap.add_argument("-k", "--samples", type=int, default=3,
                    help="teacher critiques per trajectory (Algorithm 2 K)")
    ap.add_argument("--model", default=DEFAULT_TEACHER)
    ap.add_argument("--max-requests", type=int, default=None,
                    help="cap this run's requests (default: whatever today has left)")
    ap.add_argument("--mode", default="diagnose", choices=["diagnose", "consensus"],
                    help="diagnose: verdict is given, teacher only localises "
                         "(every request yields a label). consensus: Algorithm 2 "
                         "as published, where the teacher also votes on the verdict.")
    args = ap.parse_args()

    trajectories = read_trajectories(args.trajectories)
    out = Path(args.out)
    done = {s.qid: s for s in read_supervision(out)} if out.exists() else {}
    todo = [t for t in trajectories if t.qid not in done]
    if args.limit:
        todo = todo[: max(0, args.limit - len(done))]

    budget = Budget.load(args.budget_file)
    if args.max_requests is not None:
        budget.limit = min(budget.limit, budget.requests + args.max_requests)

    # Only INCORRECT/UNSURE trajectories cost requests.
    needs_teacher = [t for t in todo if t.verdict.value != "CORRECT"]
    print(f"[sup] {len(trajectories)} trajectories, {len(done)} already labelled")
    print(f"[sup] {len(todo)} to do; {len(needs_teacher)} need the teacher "
          f"({len(todo) - len(needs_teacher)} derived free)")
    print(f"[sup] budget today: {budget.requests}/{budget.limit} used, "
          f"{budget.remaining} left -> about {budget.remaining // max(args.samples,1)} "
          f"teacher-labelled trajectories at k={args.samples}")

    teacher = Teacher(args.model, budget=budget)
    results: list[SupervisedCritique] = list(done.values())
    started, labelled, stopped = time.time(), 0, None

    for i, traj in enumerate(todo, 1):
        try:
            label = (diagnose_trajectory if args.mode == 'diagnose'
                     else supervise_trajectory)
            results.append(label(traj, teacher, k=args.samples))
            labelled += 1
        except BudgetExhausted as exc:
            stopped = str(exc)
            break
        except Exception as exc:  # noqa: BLE001 - never lose completed work
            print(f"[sup] {traj.qid}: {type(exc).__name__}: {str(exc)[:90]}")
            continue

        if labelled % 10 == 0 or i == len(todo):
            write_supervision(out, results)   # checkpoint
            rate = labelled / max(time.time() - started, 1e-6)
            print(f"[sup] {len(results)} labelled  ({rate*60:.1f}/min)  "
                  f"budget {budget.requests}/{budget.limit}", flush=True)

    write_supervision(out, results)
    summary = supervision_summary(results)
    print(f"\n[sup] wrote {len(results)} reference critiques to {out}")
    print(json.dumps(summary, indent=2))
    if stopped:
        print(f"\n[sup] STOPPED: {stopped}")
        print("[sup] re-run tomorrow to continue; completed work is saved.")


if __name__ == "__main__":
    main()
