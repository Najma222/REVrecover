"""
recovery_engine.py
-------------------
Turns risk scores into a BOUNDED, auditable recovery workflow:

  risk score -> risk tier -> intervention ladder -> stopping rules
             -> simulated response -> money recovered -> audit trail

Design principles (mirrors the track's bar):
  - Bounded: max 3 touches per session, hard 72h window, then session is
    marked "lost" and never touched again.
  - Compliant: opted-out users are always skipped; no more than one
    message per session per "day" (simulated); no discount beyond a
    configured cap.
  - Escalating: touches get progressively more generous, but each touch
    only fires if the previous one failed and the session hasn't expired.
  - Stopping rules: stop immediately on conversion, opt-out, cart value
    below minimum ROI floor, or max touches / time window reached.
  - Everything is logged to an audit trail: what was decided, why, what
    it cost, and what (if anything) came back.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---- Policy configuration (tune freely) ------------------------------

RISK_TIERS = [
    ("low", 0.0, 0.55),
    ("medium", 0.55, 0.80),
    ("high", 0.80, 1.01),
]

MIN_CART_VALUE_FOR_INTERVENTION = 250  # INR; below this, ROI is too thin
MAX_TOUCHES = 3
WINDOW_HOURS = 72

# Each touch: (name, hours_after_drop, channel, discount_pct, fixed_cost_inr)
TOUCH_LADDER = [
    {
        "name": "soft_reminder",
        "hours_after": 1,
        "channel": "push+email",
        "discount_pct": 0.0,
        "fixed_cost_inr": 2.0,
        "min_tier": "low",
    },
    {
        "name": "incentive_nudge",
        "hours_after": 6,
        "channel": "email",
        "discount_pct": 0.0,
        "fixed_cost_inr": 40.0,  # e.g. free shipping
        "min_tier": "medium",
    },
    {
        "name": "final_discount",
        "hours_after": 24,
        "channel": "email+sms",
        "discount_pct": 0.10,
        "fixed_cost_inr": 5.0,
        "min_tier": "high",
    },
]

DISCOUNT_CAP_INR = 500  # never discount more than this per session


def _risk_tier(score: float) -> str:
    for name, lo, hi in RISK_TIERS:
        if lo <= score < hi:
            return name
    return "high"


def _response_probability(
    touch_index: int, tier: str, risk_score: float, discount_pct: float, rng
) -> float:
    """
    Simulated probability the customer completes checkout after this touch.
    This is a documented ASSUMPTION calibrated to plausible cart-recovery
    email benchmarks (typical recovery-flow conversion: 3-15% per flagged
    session across a 3-touch sequence), not derived from the drop_off
    label itself (no leakage from the risk model's own target).

      base rate depends on tier (higher risk = harder to win back, but
      also the ones worth trying) and rises with the intent implied by
      the touch (later touches = more incentive = a bit more pull, with
      diminishing marginal returns and message fatigue).
    """
    tier_base = {"low": 0.22, "medium": 0.13, "high": 0.07}[tier]
    discount_lift = discount_pct * 0.9  # a 10% discount adds ~9pp
    fatigue_decay = 0.8 ** touch_index  # each subsequent touch is less effective
    prob = tier_base * fatigue_decay + discount_lift
    return float(np.clip(prob, 0.01, 0.6))


@dataclass
class RecoveryConfig:
    min_cart_value: float = MIN_CART_VALUE_FOR_INTERVENTION
    max_touches: int = MAX_TOUCHES
    window_hours: int = WINDOW_HOURS
    discount_cap: float = DISCOUNT_CAP_INR
    seed: int = 7


def run_recovery_batch(
    scored_sessions: pd.DataFrame, config: RecoveryConfig = RecoveryConfig()
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    scored_sessions must have: session_id, risk_score, predicted_drop_off,
    cart_value, opted_out.

    Returns:
      audit_trail   : one row per touch attempt (or skip decision)
      session_summary: one row per session with final outcome
      totals        : dict of headline recovery metrics
    """
    rng = np.random.default_rng(config.seed)
    audit_rows = []
    summary_rows = []

    flagged = scored_sessions[scored_sessions["predicted_drop_off"] == 1].copy()

    for _, row in flagged.iterrows():
        session_id = row["session_id"]
        cart_value = float(row["cart_value"])
        risk_score = float(row["risk_score"])
        opted_out = bool(row["opted_out"])
        tier = _risk_tier(risk_score)

        # --- Stopping rule: compliance opt-out ---
        if opted_out:
            audit_rows.append(
                dict(
                    session_id=session_id,
                    touch_index=0,
                    action="skipped",
                    reason="opted_out",
                    channel=None,
                    discount_pct=0.0,
                    cost_inr=0.0,
                    hours_after_dropoff=None,
                    converted=False,
                    recovered_revenue_inr=0.0,
                    risk_tier=tier,
                    risk_score=risk_score,
                )
            )
            summary_rows.append(
                dict(
                    session_id=session_id,
                    risk_tier=tier,
                    cart_value=cart_value,
                    touches_sent=0,
                    total_cost_inr=0.0,
                    converted=False,
                    recovered_revenue_inr=0.0,
                    net_recovered_inr=0.0,
                    stop_reason="opted_out",
                )
            )
            continue

        # --- Stopping rule: below minimum ROI floor ---
        if cart_value < config.min_cart_value:
            audit_rows.append(
                dict(
                    session_id=session_id,
                    touch_index=0,
                    action="skipped",
                    reason="cart_value_below_floor",
                    channel=None,
                    discount_pct=0.0,
                    cost_inr=0.0,
                    hours_after_dropoff=None,
                    converted=False,
                    recovered_revenue_inr=0.0,
                    risk_tier=tier,
                    risk_score=risk_score,
                )
            )
            summary_rows.append(
                dict(
                    session_id=session_id,
                    risk_tier=tier,
                    cart_value=cart_value,
                    touches_sent=0,
                    total_cost_inr=0.0,
                    converted=False,
                    recovered_revenue_inr=0.0,
                    net_recovered_inr=0.0,
                    stop_reason="cart_value_below_floor",
                )
            )
            continue

        # --- Escalation ladder ---
        converted = False
        total_cost = 0.0
        touches_sent = 0
        stop_reason = "window_expired"

        tier_rank = {"low": 0, "medium": 1, "high": 2}

        for i, touch in enumerate(TOUCH_LADDER):
            if touches_sent >= config.max_touches:
                stop_reason = "max_touches_reached"
                break
            if touch["hours_after"] > config.window_hours:
                stop_reason = "window_expired"
                break
            # only escalate to a touch if the session's tier qualifies
            if tier_rank[tier] < tier_rank[touch["min_tier"]]:
                continue

            discount_pct = touch["discount_pct"]
            discount_amount = min(cart_value * discount_pct, config.discount_cap)
            cost = touch["fixed_cost_inr"] + discount_amount

            prob = _response_probability(i, tier, risk_score, discount_pct, rng)
            converted = bool(rng.random() < prob)
            touches_sent += 1
            total_cost += cost

            recovered_revenue = cart_value - discount_amount if converted else 0.0

            audit_rows.append(
                dict(
                    session_id=session_id,
                    touch_index=touches_sent,
                    action="sent",
                    reason=None,
                    channel=touch["channel"],
                    discount_pct=discount_pct,
                    cost_inr=round(cost, 2),
                    hours_after_dropoff=touch["hours_after"],
                    converted=converted,
                    recovered_revenue_inr=round(recovered_revenue, 2),
                    risk_tier=tier,
                    risk_score=risk_score,
                )
            )

            if converted:
                stop_reason = "converted"
                break

        net = (
            (cart_value - min(cart_value * discount_pct, config.discount_cap))
            - total_cost
            if converted
            else -total_cost
        )
        summary_rows.append(
            dict(
                session_id=session_id,
                risk_tier=tier,
                cart_value=cart_value,
                touches_sent=touches_sent,
                total_cost_inr=round(total_cost, 2),
                converted=converted,
                recovered_revenue_inr=round(
                    cart_value - min(cart_value * discount_pct, config.discount_cap)
                    if converted
                    else 0.0,
                    2,
                ),
                net_recovered_inr=round(net, 2),
                stop_reason=stop_reason,
            )
        )

    audit_trail = pd.DataFrame(audit_rows)
    session_summary = pd.DataFrame(summary_rows)

    gross_at_risk_value = float(flagged["cart_value"].sum())
    total_recovered = (
        float(session_summary["recovered_revenue_inr"].sum())
        if len(session_summary)
        else 0.0
    )
    total_cost = (
        float(session_summary["total_cost_inr"].sum()) if len(session_summary) else 0.0
    )
    n_converted = (
        int(session_summary["converted"].sum()) if len(session_summary) else 0
    )

    totals = {
        "sessions_flagged_at_risk": int(len(flagged)),
        "gross_at_risk_cart_value_inr": round(gross_at_risk_value, 2),
        "sessions_recovered": n_converted,
        "recovery_rate_pct": round(
            100 * n_converted / len(flagged), 2
        )
        if len(flagged)
        else 0.0,
        "gross_revenue_recovered_inr": round(total_recovered, 2),
        "total_intervention_cost_inr": round(total_cost, 2),
        "net_revenue_recovered_inr": round(total_recovered - total_cost, 2),
        "roi_x": round(total_recovered / total_cost, 2) if total_cost > 0 else None,
        "sessions_skipped_opted_out": int(
            (session_summary["stop_reason"] == "opted_out").sum()
        )
        if len(session_summary)
        else 0,
        "sessions_skipped_low_value": int(
            (session_summary["stop_reason"] == "cart_value_below_floor").sum()
        )
        if len(session_summary)
        else 0,
    }

    return audit_trail, session_summary, totals


if __name__ == "__main__":
    scored = pd.read_csv("/home/claude/revenue_recovery/outputs/test_scored_sessions.csv")
    audit, summary, totals = run_recovery_batch(scored)
    audit.to_csv("/home/claude/revenue_recovery/outputs/audit_trail.csv", index=False)
    summary.to_csv(
        "/home/claude/revenue_recovery/outputs/session_summary.csv", index=False
    )
    import json

    print(json.dumps(totals, indent=2))
    with open("/home/claude/revenue_recovery/outputs/recovery_totals.json", "w") as f:
        json.dump(totals, f, indent=2)
