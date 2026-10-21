"""Tests for the GRPO internals.

These run on CPU in seconds and guard against the failure mode that costs the
most: a sign or normalisation error in the advantage, which does not crash,
does not warn, and simply trains on noise for hours.

Run: `python tests/test_critic_grpo.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.critic.grpo import GRPOConfig, group_advantages  # noqa: E402


# ------------------------------------------------------------ advantages ----
def test_advantages_are_zero_centred_and_unit_scale():
    adv = group_advantages([2.0, 1.0, 0.0, -1.0])
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-3


def test_better_samples_get_higher_advantage():
    """The ordering of rewards must survive into the gradient weights."""
    rewards = [-1.0, 0.2, 1.5, 0.9]
    adv = group_advantages(rewards).tolist()
    assert sorted(range(4), key=lambda i: rewards[i]) == sorted(range(4), key=lambda i: adv[i])
    # the best sample is pushed up, the worst pushed down
    assert adv[rewards.index(max(rewards))] > 0
    assert adv[rewards.index(min(rewards))] < 0


def test_flat_group_yields_exactly_zero():
    """The format-cliff case: every sample scored -1.0, so there is nothing to learn.

    Without the guard, (r - mean) is float noise and dividing by a near-zero
    std inflates it into a large bogus gradient.
    """
    for flat in ([-1.0] * 4, [0.0] * 8, [0.7] * 2):
        adv = group_advantages(flat)
        assert torch.all(adv == 0.0), f"{flat} should give no gradient"


def test_advantage_is_relative_not_absolute():
    """A group's absolute reward level must not matter - only ordering within it.

    This is why GRPO needs no value network: prompt difficulty cancels, so a
    uniformly hard prompt still produces a usable learning signal.
    """
    easy = group_advantages([0.9, 0.8, 0.7, 0.6])
    hard = group_advantages([-0.6, -0.7, -0.8, -0.9])
    assert torch.allclose(easy, hard, atol=1e-5)


def test_single_sample_group_has_no_signal():
    assert torch.all(group_advantages([0.5]) == 0.0)


# ------------------------------------------------------- loss mechanics ----
def test_policy_gradient_pushes_in_the_right_direction():
    """Reproduce the loss arithmetic and check the gradient's sign.

    A positive advantage must *increase* the log-probability of the tokens
    that earned it, and a negative advantage must decrease it. Getting this
    backwards would train the critic to do the opposite of what the rewards
    ask, while every logged number still looked plausible.
    """
    cfg = GRPOConfig()
    for advantage, expect_increase in ((1.0, True), (-1.0, False)):
        logprob = torch.tensor([-0.7, -0.4], requires_grad=True)
        old = logprob.detach()
        ratio = torch.exp(logprob - old)
        pg = -torch.min(ratio * advantage,
                        torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantage)
        pg.mean().backward()
        # gradient descent moves params by -grad, so a negative grad raises logprob
        moved_up = (-logprob.grad > 0).all().item()
        assert moved_up == expect_increase, f"advantage {advantage} moved the wrong way"


def test_ratio_is_one_on_policy():
    """One update per rollout means old == new, so the ratio starts at 1.

    The clipped objective is then a no-op in *value*, while the gradient still
    flows through new_lp - which is what makes this a policy gradient rather
    than a constant.
    """
    logprob = torch.tensor([-1.2, -0.3], requires_grad=True)
    ratio = torch.exp(logprob - logprob.detach())
    assert torch.allclose(ratio, torch.ones_like(ratio))
    ratio.sum().backward()
    assert logprob.grad is not None and torch.all(logprob.grad != 0)


def test_k3_kl_is_non_negative_and_zero_at_identity():
    """Schulman's k3 estimator: exp(d) - d - 1, with d = ref - new."""
    def k3(ref, new):
        d = ref - new
        return torch.exp(d) - d - 1.0

    same = torch.tensor([-0.5, -1.0])
    assert torch.allclose(k3(same, same), torch.zeros_like(same), atol=1e-6)
    for ref, new in ((torch.tensor([-0.5]), torch.tensor([-2.0])),
                     (torch.tensor([-2.0]), torch.tensor([-0.5]))):
        assert (k3(ref, new) > 0).all(), "divergence in either direction must be penalised"
    # and it grows as the policies separate
    near = k3(torch.tensor([-1.0]), torch.tensor([-1.2]))
    far = k3(torch.tensor([-1.0]), torch.tensor([-3.0]))
    assert far > near


def test_kl_clamp_bounds_the_exponential_tail():
    """exp() makes k3 explode; a smoke run hit KL=70 from one rare token.

    Clamping d keeps ordinary divergence intact but stops a single token from
    dominating the batch gradient.
    """
    def k3_clamped(ref, new):
        d = torch.clamp(ref - new, -10.0, 10.0)
        return torch.exp(d) - d - 1.0

    extreme = k3_clamped(torch.tensor([5.0]), torch.tensor([-40.0]))
    assert torch.isfinite(extreme).all()
    assert extreme.item() < torch.exp(torch.tensor(10.0)).item()
    # ordinary divergences are untouched by the clamp
    ordinary = torch.tensor([-1.0]), torch.tensor([-2.5])
    d = ordinary[0] - ordinary[1]
    assert torch.allclose(k3_clamped(*ordinary), torch.exp(d) - d - 1.0)


def test_inner_epochs_must_be_one():
    """Guard: >1 would silently repeat the same gradient, not reuse data."""
    from src.critic.grpo import GRPOTrainer
    try:
        GRPOTrainer(model=None, tokenizer=None, reward_fn=lambda m, t: 0.0,
                    cfg=GRPOConfig(inner_epochs=2))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "inner_epochs" in str(exc)
    # also guards a group too small to have a baseline
    try:
        GRPOTrainer(model=None, tokenizer=None, reward_fn=lambda m, t: 0.0,
                    cfg=GRPOConfig(group_size=1))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "group_size" in str(exc)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)-len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
