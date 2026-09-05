# PRD — REVrecover (Checkout Drop-off Recovery Agent)

**Track:** AI Revenue Recovery
**Direction:** Checkout drop-off recovery
**Build window:** 12 hours

---

## 1. Problem

Merchants lose revenue when a shopper adds items to a cart, browses with real
intent, and then leaves without paying. Most of that loss is silent — no
alert fires, no one follows up, and the cart is simply gone. Generic
"abandoned cart" email blasts (send everyone a coupon) waste discount budget
on shoppers who would have converted anyway, and under-invest in the
high-risk shoppers who need a stronger nudge.

## 2. What we're building

An agent that, for every checkout-initiated session:

1. **Detects** the probability the session will drop off before completing
   checkout, using behavioural signals available in real time (page
   engagement, exit/bounce rate, page value, visitor type, timing).
2. **Diagnoses** how urgent/valuable the case is (risk tier + cart value).
3. **Chooses an intervention** from a bounded, pre-approved escalation
   ladder — never an unbounded or ad-hoc action.
4. **Executes** the sequence with hard stopping rules (conversion, opt-out,
   max touches, time window, minimum cart value).
5. **Reports** exactly how much revenue was recovered, net of the cost of
   recovering it, with a full audit trail of every decision.

## 3. Non-goals (12-hour scope cut)

- No real payment gateway / real email-sending integration — the "send" step
  is simulated and logged, not actually dispatched. Swapping in a real ESP
  (SendGrid, etc.) is a follow-up, not a blocker for the demo.
- No multi-armed bandit / online learning of the response model — the
  response-probability function is a documented, static assumption (see
  §6), calibrated to plausible industry benchmarks, not fit from real
  A/B data (we don't have any labelled recovery-response data available).
- No per-user identity resolution across sessions/devices.

## 4. Data

Public dataset: **UCI "Online Shoppers Purchasing Intention" dataset**
(12,330 real e-commerce sessions, CC-BY-4.0, Sakar & Kastro 2018). It has no
native cart-value or recovery-outcome fields, so we augment it:

| Field | Source |
|---|---|
| Behavioural features (14 numeric + 3 categorical) | Native to the dataset |
| `checkout_started` | Derived: `ProductRelated >= 3` and `ProductRelated_Duration > 30` (proxy for "reached checkout") |
| `cart_value` (₹) | Synthetic, log-normal, correlated with `PageValues` and `ProductRelated_Duration` — documented assumption, not real order data |
| `opted_out` | Synthetic, ~8% of users, for the compliance stopping rule |
| `drop_off` (label) | Derived: `1 - Revenue` (Revenue = native "did this session convert") |

**Environment note:** the sandbox this was built in has no internet access,
so the pipeline (`src/data_prep.py`) tries the real UCI fetch first
(`ucimlrepo` package) and falls back to a calibrated synthetic generator that
matches the published summary statistics (12,330 rows, 15.5% positive rate)
if the fetch fails. On any machine with internet access — including your
laptop or the deployment target — it will pull the real dataset
automatically. No code change needed.

## 5. Risk model

- **Model:** Gradient Boosting Classifier (scikit-learn), trained only on
  `checkout_started == True` sessions (the population we actually act on).
- **Split:** stratified 75/25 train/test. Metrics reported are on the held-out
  test set only.
- **Threshold selection:** instead of optimizing F1/accuracy blindly, we
  sweep thresholds and pick the one that maximizes **expected net value**:
  `TP × avg_cart_value × recoverable_share − FP × false_positive_cost`.
  This directly encodes the false-positive cost the brief asks for
  (unnecessary discount/message spend on a session that would have
  converted anyway).
- **Reported metrics:** precision, recall, ROC-AUC, PR-AUC, full confusion
  matrix, and the estimated ₹ wasted on false positives at the chosen
  threshold. See `outputs/risk_metrics.json` after running.

## 6. Recovery workflow (the "agent" logic)

**Escalation ladder** (bounded — max 3 touches, 72h window):

| Touch | Timing | Channel | Incentive | Eligible tier |
|---|---|---|---|---|
| Soft reminder | T+1h | push+email | none | all flagged |
| Incentive nudge | T+6h | email | free shipping (₹40 cost) | medium/high |
| Final discount | T+24h | email+sms | 10% off (capped ₹500) | high only |

**Stopping rules (compliance + ROI):**
- Opted-out users are always skipped (0 touches).
- Carts below a minimum value floor (default ₹250) are skipped — not worth
  the intervention cost.
- Sequence stops the moment the session converts.
- Hard cap of 3 touches and a 72-hour window regardless of outcome.
- Discount is capped at ₹500/session even for a large cart.

**Response simulation (documented assumption, not derived from the risk
label):** each touch has a probability of converting the session, based on
risk tier (harder-to-win-back tiers get a lower base rate), touch number
(diminishing returns / message fatigue), and discount size. This is a
stand-in for real historical recovery-flow response data — calibrated to
publicly reported cart-recovery-email benchmarks (roughly 5–15% conversion
across a 3-touch sequence). **This is the honesty caveat**: without real
A/B data, this number is an assumption, not a measurement — but it's
transparent, and every parameter is exposed and adjustable in the Streamlit
sidebar.

## 7. What "the bar" looks like in this build

- **Honest metrics:** precision/recall/ROC-AUC/PR-AUC on a true held-out
  test set, confusion matrix, and explicit false-positive cost in ₹.
- **Measured money recovered:** gross recovered revenue, intervention cost,
  net recovered revenue, ROI multiple, all computed on the same held-out
  batch the risk model was scored on.
- **Compliant escalation:** opt-outs always respected; bounded touches;
  bounded discount; bounded time window.
- **Audit trail:** every session gets a row per touch attempt (or skip
  decision), with cost, channel, timing, and outcome — exportable as CSV.

## 8. Deliverables

- `src/data_prep.py` — dataset loading + augmentation
- `src/risk_model.py` — training, threshold selection, evaluation
- `src/recovery_engine.py` — escalation ladder, stopping rules, simulation
- `app.py` — Streamlit dashboard tying it together
- `README.md` — how to run locally and deploy
- `requirements.txt`
