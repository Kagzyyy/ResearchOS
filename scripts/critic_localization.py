"""Localisation benchmark: can a critic find WHERE a wrong answer went wrong?

    python scripts/critic_localization.py --arms base sft dqa groq

The ablation showed the critic's two jobs separate cleanly. Deciding *whether*
an answer is wrong measured AUC 0.447 - no signal at 1.5B. Saying *where* it
went wrong, given that it is wrong, beat chance by 2x. This script measures the
second job directly and compares every arm on identical inputs.

Protocol, identical for every arm:
  * held-out HotpotQA trajectories (never seen in training) whose answer is
    wrong and whose gold supporting documents are known
  * the verdict is supplied - "<verdict>INCORRECT</verdict><location>" is
    pre-filled - so only localisation is being tested, not judgement
  * a hit = the critic names `information:DocK` where DocK is one of the two
    documents that actually contain the answer
  * greedy decoding for local models; temperature 0 for the API teacher

Reported: hit rate with a Wilson 95% CI, the random baseline, and an exact
McNemar test between every pair of arms. McNemar rather than comparing two
proportions because the arms answer the *same* 100 questions - a paired test
removes question difficulty and is far more sensitive at this sample size.

Each arm's raw outputs are cached, so a crash or a Groq rate limit costs only
the arm in progress.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import src.config  # noqa: F401,E402  - loads .env for the Groq arm

from src.critic.prompts import CRITIC_INSTRUCTION, CRITIC_SYSTEM  # noqa: E402
from src.critic.schema import ParseError, Stage, Verdict, parse_critique  # noqa: E402
from src.critic.trajectory import Trajectory, read_trajectories  # noqa: E402

PREFIX = "<verdict>INCORRECT</verdict><location>"


# --------------------------------------------------------------- scoring ----
def score(traj: Trajectory, text: str) -> dict:
    """Parse one critique and decide whether it localised the error."""
    out = {"valid_format": False, "stage": None, "index": None, "hit": False}
    try:
        c = parse_critique(text, strict=True)
        out["valid_format"] = True
    except ParseError:
        try:
            c = parse_critique(text, strict=False)
        except ParseError:
            return out
    if c.location is None:
        return out
    out["stage"] = c.location.stage.value
    out["index"] = c.location.index
    out["hit"] = (c.location.stage is Stage.INFORMATION
                  and c.location.index in traj.gold_indices)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def mcnemar_exact(a_hits: list[bool], b_hits: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired outcomes.

    Only discordant pairs carry information: b = A right & B wrong, c = the
    reverse. Under H0 each discordant pair is a fair coin, so the p-value is
    a binomial tail on min(b, c) out of b + c.
    """
    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


# ------------------------------------------------------------ local arms ----
def run_local(arm: str, adapter: str | None, base: str, trajs: list[Trajectory],
              max_passage_chars: int) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.critic.local_llm import release

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    tok.truncation_side = "left"     # never cut the assistant cue off the end
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": CRITIC_SYSTEM},
             {"role": "user", "content": f"{CRITIC_INSTRUCTION}\n\n{t.render(max_passage_chars)}"}],
            tokenize=False, add_generation_prompt=True) + PREFIX
        for t in trajs
    ]
    outs: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), 4):
        enc = tok(prompts[i:i + 4], return_tensors="pt", padding=True, truncation=True,
                  max_length=1792).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=90, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 stop_strings=["</fix>"], tokenizer=tok)
        outs.extend(PREFIX + s for s in tok.batch_decode(
            gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"    {arm}: {len(outs)}/{len(prompts)}  "
              f"({len(outs) / max(time.time() - t0, 1e-6) * 60:.0f}/min)", flush=True)
    del model
    release()
    return outs


# -------------------------------------------------------------- API arm ----
GROQ_ARMS = {"groq": "openai/gpt-oss-120b", "groq20": "openai/gpt-oss-20b"}


def run_groq(trajs: list[Trajectory], max_passage_chars: int, cache: dict,
             model: str = GROQ_ARMS["groq"]) -> list[str]:
    """An API model under identical conditions: same prompt, verdict pre-filled."""
    from src.critic.supervise import Teacher

    teacher = Teacher(model=model)
    outs = []
    for i, t in enumerate(trajs, 1):
        if t.qid in cache:
            outs.append(cache[t.qid])
            continue
        msgs = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content":
                f"{CRITIC_INSTRUCTION}\n\n{t.render(max_passage_chars)}\n\n"
                "This answer is INCORRECT. Your output must begin with "
                f"{PREFIX} and continue from there."},
        ]
        try:
            text = teacher.complete(msgs, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            # One dropped TLS connection killed the first run at 42/100. Skip
            # the item without caching it, so a re-run fills exactly the gaps.
            print(f"    groq: {t.qid} failed ({type(exc).__name__}); will retry next run")
            time.sleep(5)
            continue
        # Normalise: the API model sometimes restates the prefix, sometimes not.
        if not text.lstrip().startswith("<verdict>"):
            text = PREFIX + text
        cache[t.qid] = text
        outs.append(text)
        if i % 10 == 0:
            print(f"    groq: {i}/{len(trajs)}", flush=True)
            cache_path = getattr(run_groq, "cache_path", None)
            if cache_path is not None:   # checkpoint so a crash loses <= 10 items
                cache_path.write_text(json.dumps(run_groq.raw, ensure_ascii=False),
                                      encoding="utf-8")
    return outs


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["base", "sft", "dqa", "groq20", "groq"])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft", default="models/critic/sft")
    ap.add_argument("--dqa", default="models/critic/dqa_step100")
    ap.add_argument("--trajectories", default="data/critic/trajectories_eval.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-passage-chars", type=int, default=450)
    ap.add_argument("--out", default="results/critic_localization.json")
    args = ap.parse_args()

    trajs = [t for t in read_trajectories(args.trajectories)
             if t.verdict is Verdict.INCORRECT and t.gold_indices][: args.n]
    baseline = sum(len(t.gold_indices) / len(t.passages) for t in trajs) / len(trajs)
    print(f"[loc] {len(trajs)} held-out wrong answers with known gold documents")
    print(f"[loc] random-guess baseline: {baseline:.1%}\n")

    out_path = Path(args.out)
    cache_path = out_path.with_suffix(".raw.json")
    raw: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    adapters = {"base": None, "sft": args.sft, "dqa": args.dqa}
    for arm in args.arms:
        done = raw.get(arm, {})
        if all(t.qid in done for t in trajs):
            print(f"[loc] {arm}: cached")
            continue
        print(f"[loc] running {arm}")
        if arm in GROQ_ARMS:
            done = raw.setdefault(arm, {})
            run_groq.cache_path, run_groq.raw = cache_path, raw
            try:
                run_groq(trajs, args.max_passage_chars, done, model=GROQ_ARMS[arm])
            finally:
                cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        else:
            if arm != "base" and not Path(adapters[arm]).exists():
                print(f"[loc] skipping {arm}: {adapters[arm]} missing")
                continue
            outs = run_local(arm, adapters[arm], args.base, trajs, args.max_passage_chars)
            raw[arm] = {t.qid: o for t, o in zip(trajs, outs)}
            cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # ---- score ----
    results: dict = {"n": len(trajs), "baseline": round(baseline, 4), "arms": {}}
    hits_by_arm: dict[str, list[bool]] = {}
    for arm in args.arms:
        if arm not in raw:
            continue
        scored = [score(t, raw[arm].get(t.qid, "")) for t in trajs]
        hits = [s["hit"] for s in scored]
        hits_by_arm[arm] = hits
        k = sum(hits)
        lo, hi = wilson(k, len(trajs))
        stages: dict[str, int] = {}
        for s in scored:
            if s["stage"]:
                stages[s["stage"]] = stages.get(s["stage"], 0) + 1
        results["arms"][arm] = {
            "hits": k, "hit_rate": round(k / len(trajs), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "lift_over_chance": round((k / len(trajs)) / baseline, 2),
            "valid_format": round(sum(s["valid_format"] for s in scored) / len(trajs), 4),
            "stages": stages,
        }

    arms = list(hits_by_arm)
    results["pairwise"] = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            only_a, only_b, p = mcnemar_exact(hits_by_arm[a], hits_by_arm[b])
            results["pairwise"][f"{a}_vs_{b}"] = {
                "only_" + a: only_a, "only_" + b: only_b, "p_exact": round(p, 5)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'arm':<8}{'hit rate':>10}{'95% CI':>18}{'x chance':>10}{'format':>9}")
    print("-" * 55)
    print(f"{'random':<8}{baseline:>10.1%}{'':>18}{1.0:>10.2f}")
    for arm, r in results["arms"].items():
        print(f"{arm:<8}{r['hit_rate']:>10.1%}   [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]"
              f"{r['lift_over_chance']:>10.2f}{r['valid_format']:>9.0%}")
    print("\npaired comparisons (exact McNemar):")
    for name, r in results["pairwise"].items():
        vals = list(r.values())
        print(f"  {name:<16} discordant {vals[0]}/{vals[1]}   p = {r['p_exact']:.4f}")
    print(f"\n[loc] saved -> {out_path}")


if __name__ == "__main__":
    main()
