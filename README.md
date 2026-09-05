# REVrecover — Checkout Drop-off Recovery Agent

AI Revenue Recovery track — detects checkout drop-off risk, runs a bounded
multi-touch recovery workflow, and reports measured money recovered with a
full audit trail. See `PRD.md` for the full spec and design rationale.

## Objective

Recover the revenue that quietly leaks out of abandoned checkouts. For every
session that reaches checkout, REVrecover scores abandonment risk, chases
only the carts worth chasing (risk tier + cart value) through a bounded,
pre-approved 3-touch ladder, always skips opt-outs and below-floor carts, and
reports the money actually recovered — **net of what the recovery cost** — on
a held-out test batch it has never seen.

## The problem we ran into — and how we solved it

**The false-positive money leak.** The naive fix for cart abandonment is to
send everyone a coupon — which spends discount budget on shoppers who would
have bought anyway (usually the majority). REVrecover prices that waste in
explicitly: instead of optimising raw accuracy, it sweeps risk thresholds and
picks the one that maximises **expected net value**:

`TP × cart value × recoverable share − FP × false-positive cost`

and exposes the false-positive cost as a live, tunable dial in the sidebar.

Two more we hit during the build (named in the UI, neither hidden):

- **No labelled revenue data.** The UCI dataset has no cart values, no
  checkout-start flag, and no recovery outcomes — we derive documented
  proxies (behavioural "checkout started" rule, log-normal cart values) and
  simulate response rates calibrated to published cart-recovery benchmarks.
- **Offline sandbox.** The real dataset fetch fails without internet, so the
  pipeline falls back to a calibrated synthetic twin and labels the data
  source transparently in the header.

## Project layout

```
revenue_recovery/
├── PRD.md
├── README.md              (this file)
├── requirements.txt
├── app.py                 # Streamlit dashboard (main entry point)
├── src/
│   ├── data_prep.py       # loads/augments the UCI dataset
│   ├── risk_model.py      # trains + evaluates the drop-off risk model
│   └── recovery_engine.py # escalation ladder + stopping rules + simulation
├── models/                # saved model artifacts (created on run)
├── data/                  # cached dataset (created on run)
└── outputs/               # metrics.json, audit_trail.csv, etc. (created on run)
```

## 1. Run it locally (recommended — 5 minutes)

```bash
cd revenue_recovery
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser tab with the dashboard. On first run it will try to
fetch the real UCI dataset (needs internet); if that fails it falls back
to a clearly-labelled calibrated synthetic dataset automatically — no
crash, no manual step.

## 2. Run the pipeline from the command line (no UI)

Useful for judging / CI or if you just want the raw numbers:

```bash
cd revenue_recovery/src
python3 data_prep.py        # writes data/sessions.csv
python3 risk_model.py       # trains model, writes outputs/risk_metrics.json,
                             # outputs/test_scored_sessions.csv, models/risk_model.joblib
python3 recovery_engine.py  # runs recovery batch, writes
                             # outputs/audit_trail.csv, outputs/session_summary.csv,
                             # outputs/recovery_totals.json
```

## 3. Deploy the dashboard (free, ~10 minutes)

**Streamlit Community Cloud** (easiest):
1. Push this folder to GitHub as the public repo **`REVrecover`**:
   ```bash
   git init -b main
   git remote add origin https://github.com/<YOUR-USERNAME>/REVrecover.git
   git push -u origin main
   ```
   (Private repos also work, provided you connect the account.)
2. Go to https://share.streamlit.io → "New app" → select the `REVrecover`
   repo → branch `main` → main file path `app.py`.
3. It installs `requirements.txt` automatically and gives you a public URL.
   The app fetches the real UCI dataset at runtime (the cloud has internet).

**Render / Railway** (if you want a custom domain or more control):
1. Push to GitHub.
2. Create a new Web Service, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

Either way, the app fetches the real dataset at runtime (these platforms
have internet access), so no dataset file needs to be committed to the repo.

## 4. Tuning the demo live

The sidebar in the app lets you adjust, live, without touching code:
- False-positive cost assumption (₹)
- Assumed recoverable share of a flagged cart's value
- Minimum cart value to intervene
- Maximum discount cap per session

Changing these re-runs the threshold selection and the recovery batch
instantly, so you can show the judges how the precision/recall/ROI trade-off
moves with different business assumptions — this is deliberately exposed,
not hidden, because the "recoverable share" number is an honest assumption
(see PRD §6), not a measured fact.

## 5. Known limitations

- Cart values and the customer-response-to-recovery-message model are
  documented synthetic assumptions layered on top of a real behavioural
  dataset — there's no real recovery-flow A/B data available for this
  track, so this is the best honest substitute. All assumption values are
  exposed in the sidebar for scrutiny.
- "Checkout started" is a behavioural proxy (product-page engagement
  threshold), not a real checkout-event flag, since the dataset doesn't
  log actual checkout starts.
- No live message sending — the "send" step is logged/simulated. Swapping
  in a real ESP/SMS provider is straightforward (replace one function in
  `recovery_engine.py`) but out of scope for the 12-hour build.
