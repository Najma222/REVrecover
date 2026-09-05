"""
data_prep.py
------------
Loads the UCI "Online Shoppers Purchasing Intention" dataset (12,330 real
e-commerce sessions, CC-BY-4.0) and augments it with the fields a checkout
drop-off recovery product needs that the raw dataset doesn't have:

  - cart_value       : synthetic INR cart value, correlated with PageValues /
                        ProductRelated_Duration (richer session -> bigger cart)
  - checkout_started : whether the session reached checkout (proxy: session
                        had non-trivial ProductRelated engagement)
  - opted_out        : small fraction of users who have opted out of
                        marketing messages (compliance constraint)
  - session_id       : stable id for audit trail joins

If the machine running this has no internet access (e.g. a sandboxed
environment), we fall back to a synthetic generator that is statistically
calibrated to match the real dataset's published stats (12,330 rows,
15.5% positive class, same feature ranges). This keeps the pipeline
runnable everywhere; swap back to the real fetch the moment you have a
network connection (default behaviour already prefers the real data).
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_FALLBACK_ROWS = 12330  # match real dataset size


def _load_real_uci() -> pd.DataFrame:
    """Try to fetch the real UCI dataset. Raises on any failure."""
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo

    ds = fetch_ucirepo(id=468)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_synthetic_fallback(n=N_FALLBACK_ROWS, seed=RANDOM_SEED) -> pd.DataFrame:
    """
    Calibrated synthetic stand-in for the UCI dataset, used only when the
    real dataset can't be reached (no internet). Distributions are chosen
    to match the published summary stats of the real dataset:
    ~15.5% positive class (purchase completed), heavy right-skew on
    duration/bounce/exit features, PageValues mostly 0 with a long tail.
    """
    rng = np.random.default_rng(seed)

    administrative = rng.poisson(2.3, n)
    administrative_duration = rng.gamma(1.2, 90, n) * (administrative > 0)
    informational = rng.poisson(0.5, n)
    informational_duration = rng.gamma(1.0, 60, n) * (informational > 0)
    product_related = rng.poisson(31, n) + 1
    product_related_duration = rng.gamma(1.5, 400, n)

    bounce_rates = np.clip(rng.beta(1.5, 20, n), 0, 0.2)
    exit_rates = np.clip(bounce_rates + rng.beta(1.5, 15, n) * 0.05, 0, 0.2)

    # PageValues: mostly 0, long tail for high-intent sessions
    page_values = np.where(
        rng.random(n) < 0.62, 0.0, rng.gamma(2.0, 15, n)
    )

    special_day = rng.choice(
        [0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n, p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06]
    )
    month = rng.choice(
        ["Feb", "Mar", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], n
    )
    operating_systems = rng.integers(1, 9, n)
    browser = rng.integers(1, 14, n)
    region = rng.integers(1, 10, n)
    traffic_type = rng.integers(1, 21, n)
    visitor_type = rng.choice(
        ["Returning_Visitor", "New_Visitor", "Other"], n, p=[0.85, 0.14, 0.01]
    )
    weekend = rng.random(n) < 0.23

    # Latent "purchase intent" drives Revenue (purchase completed). Weights
    # are set so PageValues dominates (as it does in the real dataset,
    # where PageValues is by far the strongest predictor of conversion),
    # with a small noise term so the signal survives filtering/calibration.
    page_value_z = (page_values - page_values.mean()) / (page_values.std() + 1e-6)
    duration_z = (
        np.log1p(product_related_duration) - np.log1p(product_related_duration).mean()
    ) / (np.log1p(product_related_duration).std() + 1e-6)
    exit_z = (exit_rates - exit_rates.mean()) / (exit_rates.std() + 1e-6)

    raw_intent = (
        2.2 * page_value_z
        + 0.6 * duration_z
        - 0.9 * exit_z
        + 0.5 * special_day
        + 0.3 * (visitor_type == "New_Visitor")
        + rng.normal(0, 0.6, n)
    )

    # Calibrate the intercept (not the probabilities directly) via bisection
    # so the positive rate matches the real dataset's 15.5%, while fully
    # preserving the rank-ordering / signal strength of raw_intent.
    target_rate = 0.155
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(raw_intent + mid)))).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    prob_purchase = 1 / (1 + np.exp(-(raw_intent + intercept)))
    revenue = rng.random(n) < prob_purchase

    df = pd.DataFrame(
        {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
            "Revenue": revenue,
        }
    )
    return df


def load_base_dataset() -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_label). Tries real data first."""
    try:
        df = _load_real_uci()
        return df, "uci_real"
    except Exception:
        df = _load_synthetic_fallback()
        return df, "synthetic_fallback"


def build_recovery_dataset(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, str]:
    """
    Full dataset used by the app: base behavioural features + the
    checkout-recovery-specific fields (cart_value, checkout_started,
    opted_out, session_id). `drop_off` (1 = abandoned) is the label the
    risk model predicts, defined as the inverse of Revenue.
    """
    df, source = load_base_dataset()
    rng = np.random.default_rng(seed)
    n = len(df)

    df = df.reset_index(drop=True)
    df["session_id"] = [f"SESS-{i:06d}" for i in range(n)]

    # Only sessions with real product engagement are treated as having
    # reached checkout (proxy signal for "started buying something").
    df["checkout_started"] = (df["ProductRelated"] >= 3) & (
        df["ProductRelated_Duration"] > 30
    )

    # Cart value (INR): correlated with PageValues + ProductRelated engagement,
    # log-normal shape typical of order-value distributions.
    base = 300 + 40 * np.log1p(df["ProductRelated_Duration"]) + 6 * df["PageValues"]
    noise = rng.lognormal(mean=0.0, sigma=0.45, size=n)
    df["cart_value"] = np.round(np.clip(base * noise, 199, 60000), 0)

    # Compliance: ~8% of users have opted out of marketing messages.
    df["opted_out"] = rng.random(n) < 0.08

    # Label: drop_off = 1 means the session did NOT convert (abandoned).
    df["drop_off"] = (~df["Revenue"].astype(bool)).astype(int)

    return df, source


FEATURE_COLUMNS_NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]
FEATURE_COLUMNS_CATEGORICAL = ["Month", "VisitorType", "Weekend"]


if __name__ == "__main__":
    df, source = build_recovery_dataset()
    print(f"Loaded {len(df)} sessions from source={source}")
    print(df[["checkout_started", "drop_off", "cart_value", "opted_out"]].describe())
    df.to_csv("/home/claude/revenue_recovery/data/sessions.csv", index=False)
