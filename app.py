"""
Checkout Drop-off Recovery Agent — Streamlit demo

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then deploy on Streamlit Community
Cloud (share.streamlit.io) pointing at app.py. See README.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_prep import build_recovery_dataset  # noqa: E402
from risk_model import CostAssumptions, train_and_evaluate  # noqa: E402
from recovery_engine import RecoveryConfig, run_recovery_batch  # noqa: E402

st.set_page_config(page_title="REVrecover · Checkout Drop-off Recovery", layout="wide")


@st.cache_data(show_spinner=False)
def _load_data():
    return build_recovery_dataset()


@st.cache_data(show_spinner=False)
def _train(df_hash_key, fp_cost, recoverable_share):
    df, _ = _load_data()
    costs = CostAssumptions(
        false_positive_cost_per_session=fp_cost,
        avg_recoverable_share=recoverable_share,
    )
    pipe, metrics, sweep_df, test_frame, pr_curve = train_and_evaluate(df, costs)
    return metrics, sweep_df, test_frame, pr_curve


@st.cache_data(show_spinner=False)
def _recover(test_frame, min_cart_value, discount_cap):
    config = RecoveryConfig(min_cart_value=min_cart_value, discount_cap=discount_cap)
    audit, summary, totals = run_recovery_batch(test_frame, config)
    return audit, summary, totals


# =====================================================================
#  Presentation layer only — no business logic lives here.
#  Everything is namespaced under `.rr-` and the app root container,
#  so it can't leak into other pages or break default widgets.
# =====================================================================

_ICONS = {
    "layers": (
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>'
    ),
    "flag": (
        '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>'
        '<line x1="4" y1="22" x2="4" y2="15"/>'
    ),
    "trend": (
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
        '<polyline points="17 6 23 6 23 12"/>'
    ),
    "card": (
        '<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/>'
        '<line x1="1" y1="10" x2="23" y2="10"/>'
    ),
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "bars": (
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
    ),
    "percent": (
        '<line x1="19" y1="5" x2="5" y2="19"/>'
        '<circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>'
    ),
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "file": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/>'
        '<line x1="16" y1="17" x2="8" y2="17"/>'
    ),
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "refresh": (
        '<polyline points="23 4 23 10 17 10"/>'
        '<polyline points="1 20 1 14 7 14"/>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"/>'
        '<path d="M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'
    ),
    "check": '<polyline points="20 6 9 17 4 12"/>',
}


def _emit(html: str) -> None:
    st.html(html)


def _icon(name: str, size: int = 16, sw: float = 1.7) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>'
    )


def _kicker(text: str) -> str:
    return f'<div class="rr-kicker">{text}</div>'


def _heading(text: str) -> str:
    return f'<div class="rr-h">{text}</div>'


def _note(text: str) -> str:
    return f'<div class="rr-note">{text}</div>'


def _chip(text: str, tone: str = "neutral") -> str:
    return f'<span class="rr-chip rr-chip-{tone}" ><i class="rr-dot"></i>{text}</span>'


def _badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="rr-badge rr-badge-{tone}" ><i class="rr-dot"></i>{text}</span>'


def _card(label, value, icon=None, delta=None, tone="ink", note=None, bar=None,
          tooltip=None):
    tip = f' title="{tooltip}"' if tooltip else ""
    icon_html = f'<span class="rr-card-icon">{_icon(icon)}</span>' if icon else ""
    delta_html = f'<div class="rr-card-delta">{delta}</div>' if delta else ""
    note_html = f'<div class="rr-card-note">{note}</div>' if note else ""
    bar_html = (
        f'<div class="rr-bar" title="{bar["pct"]}%"><i style="width:{bar["pct"]}%"></i></div>'
        if bar
        else ""
    )
    return (
        f'<div class="rr-card rr-tone-{tone}"{tip}>'
        f'<div class="rr-card-top"><span class="rr-card-label">{label}</span>{icon_html}</div>'
        f'<div class="rr-card-value">{value}</div>'
        f'{delta_html}{note_html}{bar_html}'
        f"</div>"
    )


def _grid(html_cards, cols: int = 4) -> None:
    _emit(f'<div class="rr-grid rr-grid-{cols}" >{"".join(html_cards)}</div>')


def _confusion_matrix_html(cm) -> None:
    _emit(
        '<table class="rr-cm">'
        '<thead><tr><th></th>'
        '<th>Predicted: kept browsing</th>'
        '<th>Predicted: at risk</th></tr></thead>'
        '<tbody>'
        f'<tr><th scope="row" class="cm-side">Actually kept</th>'
        f'<td class="cm-tn">{cm["true_negative"]:,}</td>'
        f'<td class="cm-fp">{cm["false_positive"]:,}</td></tr>'
        f'<tr><th scope="row" class="cm-side">Actually dropped off</th>'
        f'<td class="cm-fn">{cm["false_negative"]:,}</td>'
        f'<td class="cm-tp">{cm["true_positive"]:,}</td></tr>'
        "</tbody></table>"
    )


def _stop_bars(stop_counts) -> str:
    if not len(stop_counts):
        return ""
    mx = int(stop_counts["count"].max())
    rows = ""
    color = {"converted": "teal", "opted_out": "gold", "cart_value_below_floor": "grey",
             "max_touches_reached": "ink", "window_expired": "ink"}
    label = {"converted": "Won back", "opted_out": "Opted out (never contacted)",
             "cart_value_below_floor": "Cart below ROI floor",
             "max_touches_reached": "Max touches reached", "window_expired": "72h window expired"}
    for _, r in stop_counts.iterrows():
        key = r["stop_reason"]
        pct = round(100 * int(r["count"]) / mx, 1)
        rows += (
            f'<div class="rr-stop">'
            f'<span class="rr-stop-label">{label.get(key, key.replace("_", " "))}</span>'
            f'<span class="rr-stop-meta">{int(r["count"]):,} sessions</span></div>'
            f'<div class="rr-bar rr-bar-lg {color.get(key, "ink")}">'
            f'<i style="width:{pct}%"></i></div>'
        )
    return f'<div class="rr-stops">{rows}</div>'


def _empty_state(title, body) -> None:
    _emit(
        f'<div class="rr-empty"><span class="rr-empty-ring">{_icon("search", 26, 1.4)}</span>'
        f'<div class="rr-empty-title">{title}</div>'
        f'<div class="rr-empty-body">{body}</div></div>'
    )


# ---- Scoped design system ----------------------------------------------
# All selectors target `stAppViewContainer` (this app's root) and `.rr-`
# classes, so nothing here can change Streamlit's defaults elsewhere.
_emit(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #22333b; --ink-soft: #5b6b72; --line: rgba(34,51,59,.10);
        --teal: #2a7468; --teal-deep: #1f5a51; --teal-soft: #e8f2ef;
        --gold: #a97b2f; --gold-deep: #8a621f; --gold-soft: #f7eeda;
        --red: #b0453e; --red-soft: #f9e9e6;
        --cream: #fbf9f4; --card: #ffffff;
        --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
        --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
    }

    div[data-testid="stAppViewContainer"] {
        font-family: var(--sans);
        color: var(--ink);
    }
    div[data-testid="stAppViewContainer"] a { color: var(--teal); }

    /* ---------- Hero ---------- */
    .rr-hero { display:flex; gap:28px; align-items:center; margin:2px 0 6px; }
    .rr-hero-copy { flex:1; min-width:0; }
    .rr-kicker {
        font: 700 11px/1 var(--sans); letter-spacing:.16em;
        text-transform:uppercase; color:var(--teal); margin:30px 0 10px;
    }
    .rr-hero .rr-kicker { margin-top:0; }
    .rr-hero-title {
        font: 800 62px/1 var(--serif); letter-spacing:-.025em;
        color:var(--ink); margin:0 0 8px;
    }
    .rr-brand-accent { color: var(--teal); }
    .rr-hero-tagline {
        font: 500 24px/1.3 var(--serif); color:var(--ink);
        margin:0 0 12px; letter-spacing:-.01em;
    }
    .rr-hero-sub {
        font: 400 15px/1.65 var(--sans); color:var(--ink-soft);
        max-width:60ch; margin:0 0 18px;
    }
    .rr-hero-meta { display:flex; flex-wrap:wrap; gap:8px; margin:0; }
    .rr-chip {
        display:inline-flex; align-items:center; gap:8px;
        font:600 12px var(--sans); color:var(--ink);
        background:#fff; border:1px solid var(--line);
        border-radius:999px; padding:7px 14px;
        box-shadow:0 1px 2px rgba(34,51,59,.04);
    }
    .rr-chip i, .rr-badge i { width:8px; height:8px; border-radius:50%; display:inline-block; }
    .rr-chip-green i { background:#37a877; }
    .rr-chip-gold  i { background:var(--gold); }
    .rr-chip-neutral i { background:#b9c3c7; }
    .rr-hero-art { width:min(360px, 42vw); height:auto; flex:none; }

    /* ---------- Section headers ---------- */
    .rr-h {
        font: 700 24px/1.2 var(--serif); color:var(--ink);
        margin:0 0 8px; letter-spacing:-.01em;
    }
    .rr-note {
        font: 400 13px/1.55 var(--sans); color:var(--ink-soft);
        font-style:italic; margin:6px 0 20px; max-width:78ch;
    }

    /* ---------- Metric cards ---------- */
    .rr-grid { display:grid; gap:14px; margin:6px 0 18px; }
    .rr-grid-4 { grid-template-columns:repeat(4, minmax(0,1fr)); }
    .rr-grid-2 { grid-template-columns:repeat(2, minmax(0,1fr)); }
    @media (max-width: 1100px) { .rr-grid-4 { grid-template-columns:repeat(2, minmax(0,1fr)); } }
    @media (max-width: 640px)  { .rr-grid-4, .rr-grid-2 { grid-template-columns:1fr; } }

    .rr-card {
        background:var(--card); border:1px solid var(--line);
        border-top:3px solid var(--teal); border-radius:16px;
        padding:16px 18px 15px;
        box-shadow:0 1px 3px rgba(34,51,59,.05);
        animation: rr-rise .4s ease both;
    }
    .rr-card:hover { box-shadow:0 10px 28px rgba(34,51,59,.12); transform:translateY(-1px); }
    @keyframes rr-rise { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;} }
    .rr-card-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .rr-card-label {
        font:700 11px/1 var(--sans); letter-spacing:.09em;
        text-transform:uppercase; color:var(--ink-soft);
    }
    .rr-card-icon { color:var(--teal); display:inline-flex; }
    .rr-card-value {
        font:700 28px/1.15 var(--serif); color:var(--ink);
        margin-top:11px; font-variant-numeric:tabular-nums; letter-spacing:-.01em;
    }
    .rr-card-delta { font:500 12.5px/1.4 var(--sans); color:var(--ink-soft); margin-top:8px; }
    .rr-card-note { font:400 12px/1.5 var(--sans); color:var(--ink-soft); margin-top:8px; }
    .rr-tone-green .rr-card-value { color:var(--teal-deep); }
    .rr-tone-gold .rr-card-value, .rr-tone-gold .rr-card-icon { color:var(--gold-deep); }
    .rr-tone-red .rr-card-value, .rr-tone-red .rr-card-icon { color:var(--red); }
    .rr-tone-gold { border-top-color:var(--gold); }
    .rr-tone-red { border-top-color:var(--red); }
    .rr-tone-green { border-top-color:var(--teal); }

    .rr-bar { height:6px; background:#e9edea; border-radius:999px; overflow:hidden; margin-top:13px; }
    .rr-bar i { display:block; height:100%; border-radius:999px;
                background:var(--teal); animation: rr-grow .9s ease; }
    .rr-bar.gold i { background:var(--gold); }
    .rr-bar.red i { background:var(--red); }
    .rr-bar.ink i { background:var(--ink); }
    .rr-bar.grey i { background:#b9c3c7; }
    @keyframes rr-grow { from{width:0;} }

    /* ---------- Badges ---------- */
    .rr-badge {
        display:inline-flex; align-items:center; gap:8px;
        font:600 12.5px var(--sans); padding:7px 13px; border-radius:999px;
        border:1px solid transparent;
    }
    .rr-badge-green { background:var(--teal-soft); color:var(--teal-deep); border-color:#cfe3db; }
    .rr-badge-gold  { background:var(--gold-soft);  color:var(--gold-deep); border-color:#ebdfc0; }
    .rr-badge-red   { background:var(--red-soft);   color:var(--red);       border-color:#efd5d1; }
    .rr-badge-neutral{ background:#eef0ee;          color:var(--ink-soft);  border-color:#dde3e0; }
    .rr-badge-green i { background:#37a877; } .rr-badge-gold i { background:var(--gold); }
    .rr-badge-red i { background:var(--red); } .rr-badge-neutral i { background:#b9c3c7; }

    /* ---------- Callout ---------- */
    .rr-callout {
        display:flex; gap:12px; align-items:flex-start;
        background:var(--red-soft); border:1px solid #efd5d1; border-left:4px solid var(--red);
        border-radius:12px; padding:14px 16px; margin:6px 0 22px;
        font:400 14px/1.6 var(--sans); color:#6b3230;
    }
    .rr-callout svg { flex:none; color:var(--red); margin-top:2px; }

    /* ---------- Confusion matrix ---------- */
    .rr-cm { width:100%; border-collapse:separate; border-spacing:0;
             border:1px solid var(--line); border-radius:14px; overflow:hidden;
             box-shadow:0 1px 3px rgba(34,51,59,.05); margin:10px 0 4px; }
    .rr-cm th, .rr-cm td { padding:15px 18px; text-align:center; font-family:var(--sans); }
    .rr-cm thead th { background:var(--ink); color:#fff; font:600 12px/1 var(--sans);
                      letter-spacing:.05em; text-transform:uppercase; }
    .rr-cm .cm-side { background:#f2f4f2; color:var(--ink-soft); width:30%;
                      font:600 12px/1.4 var(--sans); text-align:left; }
    .rr-cm td { font:700 24px var(--serif); }
    .rr-cm .cm-tn { background:var(--teal-soft); color:var(--teal-deep); }
    .rr-cm .cm-fp { background:var(--red-soft); color:var(--red); }
    .rr-cm .cm-fn { background:var(--gold-soft); color:var(--gold-deep); }
    .rr-cm .cm-tp { background:#dcefe7; color:var(--teal-deep); }

    /* ---------- Stop-reason bars ---------- */
    .rr-stops { display:grid; gap:14px; margin:10px 0 4px; }
    .rr-stop { display:flex; justify-content:space-between; align-items:baseline;
               margin-bottom:6px; }
    .rr-stop-label { font:600 13.5px var(--sans); color:var(--ink); }
    .rr-stop-meta { font:500 12.5px var(--sans); color:var(--ink-soft); }
    .rr-bar-lg { height:10px; }

    /* ---------- Empty state ---------- */
    .rr-empty {
        display:flex; flex-direction:column; align-items:center; gap:6px;
        text-align:center; padding:38px 20px; border:1.5px dashed #cdd6d3;
        border-radius:16px; background:#f7faf8; color:var(--ink-soft);
        font-family:var(--sans);
    }
    .rr-empty-ring { display:inline-flex; padding:14px; border-radius:50%;
                     background:#fff; border:1px solid var(--line); color:var(--teal); }
    .rr-empty-title { font:700 15px var(--sans); color:var(--ink); }
    .rr-empty-body { font:400 13px/1.55 var(--sans); max-width:46ch; }

    /* ---------- Native widget polish (safety: values/behavior untouched) ---------- */
    div[data-testid="stAppViewContainer"] [data-testid="stDataFrame"],
    div[data-testid="stAppViewContainer"] [data-testid="stTable"] {
        border-radius:12px; overflow:hidden; border:1px solid var(--line);
    }
    div[data-testid="stAppViewContainer"] [data-testid="stDownloadButton"] button,
    div[data-testid="stAppViewContainer"] [data-testid="stBaseButton-primary"] button {
        border-radius:10px; font-weight:600; font-family:var(--sans);
    }
    div[data-testid="stAppViewContainer"] [data-testid="stExpander"] details {
        border:1px solid var(--line); border-radius:12px; background:#fff;
    }
    div[data-testid="stAppViewContainer"] [data-testid="stExpander"] summary {
        font:600 13.5px var(--sans); color:var(--ink);
    }
    section[data-testid="stSidebar"] h2 {
        color:var(--ink); font:700 19px var(--serif);
        padding-bottom:10px; border-bottom:1px solid var(--line); margin-top:22px;
    }
    section[data-testid="stSidebar"] h3 { color:var(--ink); font:700 13.5px var(--sans); }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color:var(--ink-soft); font-size:12px; line-height:1.5;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color:var(--teal); font-weight:600;
    }
    </style>
    """
)

# =====================================================================
#  Hero
# =====================================================================
_emit(
    '<div class="rr-hero">'
    '<div class="rr-hero-copy">'
    f'{_kicker("Revenue operations · Checkout drop-off recovery")}'
    '<h1 class="rr-hero-title"><span class="rr-brand-accent">REV</span>recover</h1>'
    '<p class="rr-hero-tagline">Win back the carts that got away</p>'
    '<p class="rr-hero-sub">A risk model scores every checkout session, a bounded '
    "three-touch sequence nudges the ones about to walk away, and every rupee "
    "recovered — or spent — is audited on a held-out test batch. No guesswork: "
    'the economics are on this screen.</p>'
    '<p class="rr-hero-meta">'
    f'{_chip("Held-out test batch", "neutral")}'
    f'{_chip("3-touch ladder · 72h window", "neutral")}'
    f'{_chip("Opt-outs always skipped", "green")}'
    "</p></div>"
    '<svg class="rr-hero-art" viewBox="0 0 360 158" role="img" aria-label="Recovery at a glance">'
    "<defs>"
    '<linearGradient id="rrArea" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#2a7468" stop-opacity=".25"/>'
    '<stop offset="1" stop-color="#2a7468" stop-opacity="0"/></linearGradient>'
    '<pattern id="rrDots" width="18" height="18" patternUnits="userSpaceOnUse">'
    '<circle cx="2" cy="2" r="1.2" fill="#22333b" opacity=".06"/></pattern>'
    "</defs>"
    '<rect width="360" height="158" rx="22" fill="url(#rrDots)"/>'
    '<rect width="360" height="158" rx="22" fill="#f0f7f4"/>'
    '<rect x="16" y="16" width="172" height="38" rx="12" fill="#fff" stroke="#dFede7"/>'
    '<circle cx="34" cy="35" r="5" fill="#2a7468"/>'
    '<rect x="46" y="26" width="58" height="6" rx="3" fill="#c9d8d3"/>'
    '<rect x="46" y="36" width="90" height="6" rx="3" fill="#e6ece9"/>'
    '<path d="M20 128 C 88 108, 128 98, 168 84 S 260 48, 340 26 L 340 158 L 20 158 Z" fill="url(#rrArea)"/>'
    '<path d="M20 128 C 88 108, 128 98, 168 84 S 260 48, 340 26" fill="none" stroke="#2a7468" stroke-width="3" stroke-linecap="round"/>'
    '<circle cx="20" cy="128" r="4.5" fill="#fff" stroke="#2a7468" stroke-width="2.2"/>'
    '<circle cx="168" cy="84" r="4.5" fill="#fff" stroke="#2a7468" stroke-width="2.2"/>'
    '<circle cx="340" cy="26" r="6" fill="#2a7468"/>'
    '<rect x="224" y="98" width="118" height="38" rx="12" fill="#fff" stroke="#e7e0cf"/>'
    '<text x="283" y="122" font-family="Fraunces, Georgia, serif" font-size="15" font-weight="700" fill="#22333b" text-anchor="middle">net recovered</text>'
    "</svg></div>"
)

df, source = _load_data()
if source == "uci_real":
    _emit(
        '<div style="margin:-4px 0 2px">'
        + _chip("Canonical UCI dataset · 12,330 real sessions", "green")
        + "</div>"
    )
else:
    _emit(
        '<div style="margin:-4px 0 2px">'
        + _chip("Calibrated synthetic stand-in (statistically matched to the "
                "real UCI dataset)", "gold")
        + "</div>"
    )

# =====================================================================
#  Sidebar — policy & cost assumptions
# =====================================================================
with st.sidebar:
    st.header("Policy & cost assumptions")
    st.caption(
        "Everything here re-runs the simulation instantly. Start with the "
        "defaults — they were tuned to a sensible break-even point."
    )

    st.subheader("Risk economics")
    fp_cost = st.slider(
        "False-positive cost per session (₹)",
        min_value=10,
        max_value=300,
        value=120,
        step=10,
        help="What you pay (incentive given away + aggravation) when you nudge "
        "someone who was going to buy anyway. Higher → you flag fewer people, "
        "miss more at-risk carts.",
    )
    recoverable_share = st.slider(
        "Assumed recoverable share of a flagged cart's value",
        min_value=0.03,
        max_value=0.25,
        value=0.09,
        step=0.01,
        help="Of a correctly flagged at-risk cart, how much of its value you "
        "realistically expect the recovery messages to win back. Higher → "
        "recovery looks more worthwhile.",
    )

    st.subheader("Recovery bounds")
    min_cart_value = st.slider(
        "Minimum cart value to intervene (₹)", 0, 1000, 250, 50,
        help="Below this, a cart is too small to bother chasing — the messaging "
        "cost would eat the whole win. Raise it to skip small carts entirely.",
    )
    discount_cap = st.slider(
        "Max discount per session (₹)", 100, 1000, 500, 50,
        help="The biggest single discount we'll ever hand out to win one cart "
        "back. Raise it to win bigger carts, at the cost of giving away more.",
    )

    with st.expander("What am I actually tuning?"):
        st.caption(
            "- **False-positive cost**: how much a wasted nudge costs. Raise it "
            "to flag fewer carts (and miss more).\n"
            "- **Recoverable share**: the slice of a flagged cart's value you "
            "expect the messages to win back. Raise it and recovery looks more "
            "worthwhile.\n"
            "- **Minimum cart value**: below this, messaging costs eat the whole "
            "win, so we skip.\n"
            "- **Max discount**: the largest single incentive we'll ever give "
            "away per session."
        )

    st.divider()
    st.caption(
        "Escalation ladder (fixed and bounded): 1h soft reminder · 6h incentive "
        "nudge (medium & high risk) · 24h 10% discount (high risk only). At most "
        "3 touches inside a 72h window. Opt-outs are never contacted."
    )

with st.spinner("Training the risk model and running your recovery batch..."):
    metrics, sweep_df, test_frame, pr_curve = _train(
        source, fp_cost, recoverable_share
    )
    audit, summary, totals = _recover(
        test_frame, min_cart_value, discount_cap
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Risk model", "Recovery results", "Audit trail"]
)

# =====================================================================
#  Tab 1 · Overview
# =====================================================================
with tab1:
    _emit(
        f'<p class="rr-hero-sub" style="margin:2px 0 22px; max-width:76ch">'
        f"Headline story from the holdout: we scored <strong>{metrics['n_test_sessions']:,} "
        f"sessions</strong> the model had never seen, flagged "
        f"<strong>{totals['sessions_flagged_at_risk']:,}</strong> as worth "
        "chasing, and this is what chasing them actually bought back — "
        "and what it cost.</p>"
    )

    net_share = (
        f"{100*totals['net_revenue_recovered_inr']/totals['gross_at_risk_cart_value_inr']:.1f}% of gross at-risk value"
        if totals["gross_at_risk_cart_value_inr"]
        else None
    )
    _grid([
        _card("Sessions evaluated", f"{metrics['n_test_sessions']:,}",
              icon="layers", note="All checkout-initiated sessions held out for testing"),
        _card("Flagged at risk", f"{totals['sessions_flagged_at_risk']:,}",
              icon="flag", note="Predicted likely to abandon checkout",
              bar={"pct": round(100 * totals['sessions_flagged_at_risk'] / metrics['n_test_sessions'], 1)}),
        _card("Recovery rate", f"{totals['recovery_rate_pct']}%",
              icon="trend", tone="green",
              delta="of flagged carts won back",
              bar={"pct": totals['recovery_rate_pct']}),
        _card("Net recovered", f"₹{totals['net_revenue_recovered_inr']:,.0f}",
              icon="card", tone="green", delta=net_share,
              tooltip="Gross revenue recovered minus total intervention cost"),
    ])

    if totals["roi_x"] and totals["roi_x"] > 1:
        roi_delta = _badge(f"Profitable — every rupee brings back ₹{totals['roi_x']}",
                           "green")
    elif totals["roi_x"]:
        roi_delta = _badge("Near break-even — tighten the dials to improve", "gold")
    else:
        roi_delta = _badge("No spend this run — ROI undefined", "neutral")

    _grid([
        _card("Precision", f"{metrics['precision']:.2%}",
              icon="target", delta="of flags are true at-risk"),
        _card("Recall", f"{metrics['recall']:.2%}",
              icon="pulse", tone="green", delta="of real drop-offs caught"),
        _card("ROC–AUC", f"{metrics['roc_auc']:.3f}",
              icon="bars", note="Rank-quality of scores; 1.0 is a perfect ordering"),
        _card("ROI", f"{totals['roi_x']}x" if totals["roi_x"] else "n/a",
              icon="percent", tone="gold", delta=roi_delta),
    ])

    _emit(_kicker("Money flow"))
    _emit(_heading("From at-risk carts to net recovered"))
    seg = st.segmented_control("View", ["Chart", "Table"], default="Chart",
                               label_visibility="collapsed")
    flow_df = pd.DataFrame(
        {
            "Stage": [
                "Gross at-risk cart value",
                "Gross revenue recovered",
                "Intervention cost",
                "Net revenue recovered",
            ],
            "Amount (₹)": [
                totals["gross_at_risk_cart_value_inr"],
                totals["gross_revenue_recovered_inr"],
                -totals["total_intervention_cost_inr"],
                totals["net_revenue_recovered_inr"],
            ],
        }
    )
    if seg == "Table":
        st.dataframe(
            flow_df.style.format({"Amount (₹)": "₹{:,.0f}"}),
            width="stretch", height=260,
        )
    else:
        st.bar_chart(flow_df.set_index("Stage"), height=300)
    _emit(
        _note(
            "The full journey of a flagged cart's value — what was on the table "
            "at the start, what the nudges clawed back, what the nudges themselves "
            "cost (the bar below zero), and what's left as net recovered."
        )
    )

    _emit(_kicker("Compliance & ROI bounds"))
    _emit(_heading("Why some carts were never chased"))

    opted_badge = (
        f'{_badge("Opt-outs respected — never contacted", "gold")}'
        if totals["sessions_skipped_opted_out"] > 0
        else ""
    )
    _grid([
        _card("Skipped · opted out",
              f"{totals['sessions_skipped_opted_out']:,}",
              icon="shield", tone="gold",
              delta=opted_badge or "No opt-outs this run"),
        _card("Skipped · below ROI floor",
              f"{totals['sessions_skipped_low_value']:,}",
              icon="filter", note="Carts too small for the messaging cost to pay off"),
    ], cols=2)

    with st.expander("What went into this run — and its honest limits"):
        st.caption(
            "The model trains only on checkout-initiated sessions and never sees "
            "the test batch. The threshold is chosen to maximise expected net "
            "value, not accuracy, so a wrong nudge carries an explicit price tag. "
            "The recovery ladder and the link between spend and win-back are "
            "simulated from documented assumptions, not from the drop-off label, "
            "so there's no leakage into the risk model's own score."
        )

# =====================================================================
#  Tab 2 · Risk model
# =====================================================================
with tab2:
    _emit(
        '<p class="rr-hero-sub" style="margin:2px 0 10px; max-width:76ch">'
        "An honesty check before this model touches real money: these are its "
        "results on shoppers it has never met. The operating threshold isn't the "
        "one that looks best on a chart — it's the one that makes the most money "
        "once you price in the cost of annoying the people we're wrong about.</p>"
    )

    _grid([
        _card("Base drop-off rate (test)", f"{metrics['base_drop_off_rate_test']:.1%}",
              icon="pulse", note="Sessions that actually abandoned checkout"),
        _card("Average cart value (test)", f"₹{metrics['avg_cart_value_test_inr']:,.0f}",
              icon="file", note="Value at stake per session, on average"),
        _card("PR–AUC", f"{metrics['pr_auc']:.3f}",
              icon="target", note="Precision-recall ranking; closer to 1.0 is better"),
        _card("Operating threshold", f"{metrics['chosen_threshold']:.3f}",
              icon="percent", tone="gold",
              note="Chosen to maximise expected net value"),
    ])

    _emit(_kicker("Confusion matrix"))
    _emit(_heading("What the model gets wrong — and what that costs"))
    cm = metrics["confusion_matrix"]
    _confusion_matrix_html(cm)
    _emit(
        _note(
            "Top-right (false positives) is money wasted on shoppers who were going "
            "to buy anyway; bottom-left (false negatives) is recovered revenue "
            "quietly missed at this threshold."
        )
    )

    _emit(
        '<div class="rr-callout">'
        f'{_icon("flag", 18)}'
        f"<span><strong>{cm['false_positive']:,} false positives</strong> — flagged but "
        "would have converted anyway. At the current false-positive cost that's an "
        f"estimated <strong>₹{metrics['estimated_wasted_spend_on_false_positives_inr']:,.0f}</strong> "
        f"in unnecessary incentives and messaging (₹{fp_cost}/session)."
        "</span></div>"
    )

    _emit(_kicker("Precision vs recall"))
    _emit(_heading("The trade-off you can't dodge"))
    pr_precision, pr_recall = pr_curve
    pr_df = pd.DataFrame(
        {"Recall": pr_recall[:-1], "Precision": pr_precision[:-1]}
    ).sort_values("Recall")
    st.line_chart(pr_df.set_index("Recall"), height=300)
    _emit(
        _note(
            "Again: catch more at-risk shoppers (further right) and you inevitably "
            "tag more innocent ones (precision falls). The ideal perch is the "
            "top-right corner — hard, but the model sits close."
        )
    )

    _emit(_kicker("Threshold sweep"))
    _emit(_heading("Every threshold, scored by the money it makes"))
    st.dataframe(
        sweep_df.style.highlight_max(subset=["expected_net_value"], color="#dcefe7"),
        width="stretch", height=280,
    )
    _emit(
        _note(
            "The highlighted row is the operating point we use — not the row with "
            "the best accuracy, but the row with the best bottom line after "
            "false-positive costs."
        )
    )

# =====================================================================
#  Tab 3 · Recovery results
# =====================================================================
with tab3:
    _emit(
        '<p class="rr-hero-sub" style="margin:2px 0 10px; max-width:76ch">'
        "Zoom in on the workflow itself, split by risk tier — who we won back, "
        "who slipped away, and what each win actually cost. This is where you can "
        "see whether the expensive high-risk shoppers earn their keep.</p>"
    )

    _grid([
        _card("Sessions flagged", f"{totals['sessions_flagged_at_risk']:,}",
              icon="layers", delta="at-risk carts entering the ladder"),
        _card("Won back", f"{totals['sessions_recovered']:,}",
              icon="check", tone="green",
              delta=f"{totals['recovery_rate_pct']}% of flagged"),
        _card("Gross recovered", f"₹{totals['gross_revenue_recovered_inr']:,.0f}",
              icon="trend", tone="green", note="Before intervention costs"),
        _card("Total intervention cost", f"₹{totals['total_intervention_cost_inr']:,.0f}",
              icon="card", tone="gold", note="Fixed touch fees + discounts given"),
    ])

    if len(summary):
        by_tier = (
            summary.groupby("risk_tier")
            .agg(
                sessions=("session_id", "count"),
                converted=("converted", "sum"),
                cart_value=("cart_value", "sum"),
                recovered=("recovered_revenue_inr", "sum"),
                cost=("total_cost_inr", "sum"),
            )
            .reset_index()
        )
        by_tier["recovery_rate_%"] = (
            100 * by_tier["converted"] / by_tier["sessions"]
        ).round(1)

        _emit(_kicker("By risk tier"))
        _emit(_heading("Where the recovery happened"))
        st.dataframe(by_tier, width="stretch", height=220)
        st.bar_chart(by_tier.set_index("risk_tier")[["recovered", "cost"]], height=280)
        _emit(
            _note(
                "For each tier: revenue the nudges brought back vs. what the nudges "
                "cost. When recovered towers above cost, that tier is earning its keep."
            )
        )

        _emit(_kicker("How each chase ended"))
        _emit(_heading("Stop reasons across all flagged sessions"))
        stop_counts = summary["stop_reason"].value_counts().reset_index()
        stop_counts.columns = ["stop_reason", "count"]
        _emit(_stop_bars(stop_counts))
        _emit(_note("Only converted sessions cost recoverable; opt-outs and "
                    "below-floor carts are never contacted or charged."))
    else:
        _empty_state(
            "No sessions were flagged at these settings.",
            "No cart looked risky enough to chase. Loosen the false-positive cost or "
            "the minimum cart value and the workflow will spring to life.",
        )

    st.download_button(
        "Download session summary (CSV)",
        summary.to_csv(index=False),
        file_name="session_summary.csv",
    )

# =====================================================================
#  Tab 4 · Audit trail
# =====================================================================
with tab4:
    _emit(
        '<p class="rr-hero-sub" style="margin:2px 0 10px; max-width:76ch">'
        "The tripwire log — a line for every decision, whether the agent sent a "
        "message or deliberately held back. If anyone ever asks why we spent "
        "anything, the answer is a filter-click away.</p>"
    )

    _emit(_heading("Full audit trail — every decision, timestamped and costed"))
    _emit(
        _note(
            "Every touch attempt and every skip decision (opt-out, below ROI "
            "floor) is logged, so any recovered rupee — or any spend — can be "
            "traced to the exact rule that triggered it."
        )
    )

    col1, col2 = st.columns(2)
    tier_filter = col1.multiselect(
        "Filter by risk tier",
        options=sorted(audit["risk_tier"].dropna().unique())
        if len(audit)
        else [],
        default=None,
    )
    action_filter = col2.multiselect(
        "Filter by action",
        options=sorted(audit["action"].dropna().unique()) if len(audit) else [],
        default=None,
    )

    view = audit.copy()
    if tier_filter:
        view = view[view["risk_tier"].isin(tier_filter)]
    if action_filter:
        view = view[view["action"].isin(action_filter)]

    if len(audit):
        st.dataframe(view, width="stretch", height=400)
        _emit(
            _note(
                f"Showing {len(view):,} of {len(audit):,} log lines — use the "
                "filters above to narrow the trail."
            )
        )
    else:
        _empty_state(
            "Nothing to audit yet.",
            "No sessions were flagged at these settings, so the log is empty. "
            "Loosen the dials and the trail will fill in.",
        )

    st.download_button(
        "Download full audit trail (CSV)",
        audit.to_csv(index=False),
        file_name="audit_trail.csv",
    )

    _emit(_kicker("Drill into a single session"))
    _emit(_heading("One session's escalation timeline"))
    if len(audit):
        session_options = audit["session_id"].unique().tolist()
        chosen = st.selectbox("Session ID", session_options)
        st.table(audit[audit["session_id"] == chosen])
    else:
        _emit(_note("Select a session once the log has entries."))