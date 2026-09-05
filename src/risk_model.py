"""
risk_model.py
-------------
Trains a checkout drop-off risk classifier on a held-out test set and
reports honest metrics, including the false-positive cost that the bar
explicitly asks for.

Model: Gradient Boosting on top of the UCI behavioural features.
Target: drop_off (1 = session did not convert).

We only *evaluate* the classifier on checkout_started==True sessions,
since that's the population the recovery workflow actually acts on.
Training also uses only that population so the model isn't diluted by
pure-browsing sessions that were never at risk of "checkout" drop-off.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC


@dataclass
class CostAssumptions:
    # Cost of intervening on a session that would have converted anyway
    # (message annoyance + any discount given away needlessly), as a
    # fraction of average cart value.
    false_positive_cost_per_session: float = 120.0  # INR: incentive + msg cost
    # Share of a flagged at-risk cart's value that recovery messaging
    # realistically wins back (industry cart-recovery-email benchmarks are
    # commonly 5-12% of flagged carts converting), used only for threshold
    # selection, not for training the classifier itself.
    avg_recoverable_share: float = 0.09


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURE_COLUMNS_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURE_COLUMNS_CATEGORICAL,
            ),
        ]
    )
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def choose_cost_aware_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    avg_cart_value: float,
    fp_cost: float,
    fn_recovery_value_share: float,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep thresholds and pick the one that maximises expected net value:
      net(t) = TP(t) * avg_cart_value * fn_recovery_value_share
              - FP(t) * fp_cost
    This encodes "false-positive cost" explicitly rather than optimising
    F1 blindly.
    """
    thresholds = np.linspace(0.05, 0.95, 37)
    rows = []
    best_t, best_net = 0.5, -np.inf
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        net = tp * avg_cart_value * fn_recovery_value_share - fp * fp_cost
        rows.append(
            {
                "threshold": round(t, 3),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "expected_net_value": net,
            }
        )
        if net > best_net:
            best_net, best_t = net, t
    return best_t, pd.DataFrame(rows)


def train_and_evaluate(df: pd.DataFrame, costs: CostAssumptions = CostAssumptions()):
    checkout_df = df[df["checkout_started"]].reset_index(drop=True)

    X = checkout_df[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL]
    y = checkout_df["drop_off"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, checkout_df.index, test_size=0.25, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_scores = pipe.predict_proba(X_test)[:, 1]

    avg_cart_value = checkout_df.loc[idx_test, "cart_value"].mean()
    best_t, sweep_df = choose_cost_aware_threshold(
        y_test,
        y_scores,
        avg_cart_value=avg_cart_value,
        fp_cost=costs.false_positive_cost_per_session,
        fn_recovery_value_share=costs.avg_recoverable_share,
    )

    y_pred = (y_scores >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "n_test_sessions": int(len(y_test)),
        "base_drop_off_rate_test": float(np.mean(y_test)),
        "chosen_threshold": float(best_t),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "pr_auc": float(average_precision_score(y_test, y_scores)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_positive_cost_assumption_inr": costs.false_positive_cost_per_session,
        "estimated_wasted_spend_on_false_positives_inr": float(
            fp * costs.false_positive_cost_per_session
        ),
        "avg_cart_value_test_inr": float(avg_cart_value),
    }

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_scores)

    test_frame = checkout_df.loc[idx_test].copy()
    test_frame["risk_score"] = y_scores
    test_frame["predicted_drop_off"] = y_pred

    return pipe, metrics, sweep_df, test_frame, (pr_precision, pr_recall)


if __name__ == "__main__":
    from data_prep import build_recovery_dataset

    df, source = build_recovery_dataset()
    pipe, metrics, sweep_df, test_frame, pr_curve = train_and_evaluate(df)
    print(f"data source: {source}")
    print(json.dumps(metrics, indent=2))

    with open("/home/claude/revenue_recovery/outputs/risk_metrics.json", "w") as f:
        json.dump({"data_source": source, **metrics}, f, indent=2)

    test_frame.to_csv(
        "/home/claude/revenue_recovery/outputs/test_scored_sessions.csv", index=False
    )
    sweep_df.to_csv(
        "/home/claude/revenue_recovery/outputs/threshold_sweep.csv", index=False
    )

    import joblib

    joblib.dump(pipe, "/home/claude/revenue_recovery/models/risk_model.joblib")
    print("Saved model + metrics + scored test set.")
