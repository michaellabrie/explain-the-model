"""Shared pieces for every module: the dataset and validation-subject picker, number formats,
the page scaffold (header, numbered sections, figure and table panels, notes) and chart styling.

Every number shown anywhere in the app comes from outputs/*.json or the m10 helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import m10

LABELS = {"glucose": "Fasting glucose", "sbp": "Blood pressure", "fev1": "Lung function (FEV₁)"}
# Feature that residual plots show first. For lung function, age shows where one straight line fails.
FOCUS_FEATURE = {"glucose": "hba1c", "sbp": "age", "fev1": "age"}
# The same eight numbered sections structure every module page.
SECTIONS = ["Objective", "Analysis workflow", "Validation subject", "Feature profile", "Model output",
            "Quantitative interpretation", "Scientific interpretation", "Research translation"]

# One categorical palette with separate light and dark steps (checked for colour-blind separation).
# Roles used across the app: blue = model prediction / training data, orange = validation data or
# the highlighted item, aqua = a measured (observed) value, red/blue = upward/downward push,
# violet = the learning-rate-0.02 comparison model, ink = a "you are here" marker.
_LIGHT = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#6250d6", red="#e34948",
              muted="#898781", faint="#d3d2cc", ink="#0b0b0b", surface="#ffffff",
              axis="#b8b7b0", grid="rgba(82,81,78,0.10)")
_DARK = dict(blue="#3987e5", orange="#d95926", aqua="#199e70", violet="#9085e9", red="#e66767",
             muted="#8b949e", faint="#3a414b", ink="#e6edf3", surface="#0d1117",
             axis="#3d444d", grid="rgba(139,148,158,0.12)")

STYLE = """
<style>
.lab-eyebrow{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:.7rem;letter-spacing:.14em;
  text-transform:uppercase;color:#6fa8dc;margin:0 0 -.3rem}
.lab-question{font-size:1.04rem;opacity:.8;margin:-.35rem 0 .8rem}
dl.lab-meta{display:grid;grid-template-columns:max-content 1fr;gap:.28rem 1.4rem;margin:.2rem 0 .4rem;
  padding:.65rem 0;border-top:1px solid rgba(139,148,158,.28);border-bottom:1px solid rgba(139,148,158,.28);
  font-size:.84rem}
dl.lab-meta dt{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;color:#8b949e;padding-top:.14rem}
dl.lab-meta dd{margin:0}
.lab-sec{display:flex;gap:.8rem;align-items:baseline;margin:2.2rem 0 -.1rem;padding-top:.55rem;
  border-top:1px solid rgba(139,148,158,.28);font-family:'IBM Plex Mono',ui-monospace,monospace;
  font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;color:#8b949e}
.lab-sec .n{color:#6fa8dc}
.fig-head{margin-bottom:.15rem}
.fig-head .num{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:.68rem;letter-spacing:.12em;
  text-transform:uppercase;color:#6fa8dc;margin-right:.65rem}
.fig-head .title{font-weight:600}
.fig-head .sub{font-size:.83rem;color:#8b949e;margin-top:.12rem}
[data-testid="stMetric"]{border:1px solid rgba(139,148,158,.28);padding:.5rem .8rem .45rem;
  background:rgba(139,148,158,.05)}
[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',ui-monospace,monospace}
[data-testid="stMetricLabel"] p{font-size:.76rem;color:#8b949e}
[data-testid="stMarkdownContainer"] blockquote{border-left:2px solid #6fa8dc;background:rgba(111,168,220,.07);
  margin:.6rem 0;padding:.5rem 1rem}
[data-testid="stMarkdownContainer"] blockquote,[data-testid="stMarkdownContainer"] blockquote *{
  color:inherit!important;opacity:1!important}
[data-testid="stMarkdownContainer"] blockquote p:last-child{margin-bottom:0}
[data-testid="stCaptionContainer"]{font-size:.81rem;line-height:1.55}
.lab-brand{margin:-.4rem 0 .4rem}
.lab-brand .name{font-weight:600;font-size:1.02rem;line-height:1.25}
</style>
"""


def palette() -> dict:
    try:
        return _DARK if st.context.theme.type == "dark" else _LIGHT
    except Exception:  # no theme information (e.g. headless tests)
        return _DARK


def apply_style() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


# ---- Numbers and labels ------------------------------------------------------
def signed(v: float, d: int = 1) -> str:
    return f"{v:+.{d}f}"


def num(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}"


def val(v: float) -> str:
    """Feature values and coefficients, to 4 significant digits."""
    return f"{v:.4g}"


def unit(u: str) -> str:
    """Units to print after a feature value ('ratio' and '0/1' read better without)."""
    return "" if u in ("ratio", "0/1") else f" {u}"


def outcome(b: dict, lower: bool = False) -> str:
    """Short name of the predicted outcome, e.g. 'Fasting glucose' or 'FEV₁' (lower=True for mid-sentence use)."""
    name = b["target_label"].split(" (")[0].split(":")[0]
    return name[0].lower() + name[1:] if lower and not name.startswith("FEV") else name


def feature(b: dict, name: str) -> dict:
    return next(f for f in b["features"] if f["name"] == name)


def label_of(b: dict, name: str) -> str:
    return next((f["label"] for f in b["features"] if f["name"] == name), name)


def short_label(b: dict, name: str) -> str:
    return label_of(b, name).split(" (")[0]


def push_label(b: dict, key: str) -> str:
    """Name of one grouped push: a category ('Sex') or a numeric feature."""
    return next((g["label"] for g in b["categorical_groups"] if g["name"] == key), None) or short_label(b, key)


def grouped_columns(b: dict) -> set[str]:
    return {lv["column"] for g in b["categorical_groups"] for lv in g["levels"] if lv["column"]}


def numeric_features(b: dict) -> list[dict]:
    grouped = grouped_columns(b)
    return [f for f in b["features"] if f["name"] not in grouped]


def level_of(g: dict, values: dict) -> str:
    """The chosen level of a category group, read from its 0/1 columns."""
    return next((lv["label"] for lv in g["levels"] if lv["column"] and values[lv["column"]] == 1), g["reference"])


def column_level(b: dict, column: str) -> tuple[dict, str] | None:
    """(group, level label) for a 0/1 category column, or None for a numeric feature."""
    for g in b["categorical_groups"]:
        for lv in g["levels"]:
            if lv["column"] == column:
                return g, lv["label"]
    return None


@st.cache_data(show_spinner=False)
def load_data(key: str) -> pd.DataFrame:
    return m10.load_data(key)


# ---- Sidebar -----------------------------------------------------------------
def pick_person() -> tuple[dict, dict]:
    """Sidebar dataset and validation-subject pickers. Every module starts here."""
    st.sidebar.markdown('<div class="lab-brand"><div class="lab-eyebrow">Computational notebook</div>'
                        '<div class="name">The Researcher\'s Regression Lab</div></div>', unsafe_allow_html=True)
    key = st.sidebar.selectbox("Dataset", m10.DATASETS, format_func=LABELS.get, key="dataset")
    b = m10.load_bundle(key)
    labels = {p["id"]: f"{p['label']}: {p['description']}" for p in b["people"]}
    pid = st.sidebar.selectbox("Validation subject", list(labels), format_func=labels.get, key=f"person_{key}",
                               help="One of five real survey participants chosen in advance. The same five "
                                    "appear in every module.")
    person = next(p for p in b["people"] if p["id"] == pid)
    with st.sidebar:
        meta_grid({"Subject": f"{person['label']} ({person['description']})",
                   "Selected as": person["why_chosen"],
                   "Measured": f"{num(person['observed'])} {b['units']}",
                   "Partition": f"{person['split']} (not used for fitting)"})
        with st.expander("Dataset record", icon=":material/dataset:"):
            feats = "".join(f"\n- {f['label']} ({f['units']})" for f in numeric_features(b))
            cats = "".join(f"\n- {g['label']}: {', '.join(lv['label'] for lv in g['levels'])}"
                           for g in b["categorical_groups"])
            st.markdown(f"**{b['name']}**  \nOutcome: **{b['target_label']}** ({b['units']}). Predictors:{feats}{cats}")
            st.caption(f"{b['n_train']} training rows and {b['n_validation']} validation rows "
                       f"({b['split_description']}). Source: [{b['source']}]({b['url']}). "
                       "These describe this survey sample, not the whole US population.")
    return b, person


# ---- Page scaffold -------------------------------------------------------------
def meta_grid(items: dict) -> None:
    rows = "".join(f"<dt>{escape(str(k))}</dt><dd>{escape(str(v))}</dd>" for k, v in items.items())
    st.markdown(f'<dl class="lab-meta">{rows}</dl>', unsafe_allow_html=True)


def page_header(b: dict, module: int, name: str, model: str, title: str, question: str, spec: str) -> None:
    st.markdown(f'<div class="lab-eyebrow">Module {module} · {escape(name)} · {escape(model)}</div>',
                unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<div class="lab-question">{escape(question)}</div>', unsafe_allow_html=True)
    meta_grid({"Dataset": b["name"], "Outcome": f"{b['target_label']} ({b['units']})",
               "Partition": f"{b['n_train']} training · {b['n_validation']} validation · fixed 80/20 split, seed 0",
               "Model": spec, "Source": b["source"]})


def rule(tag: str, label: str) -> None:
    """A numbered section rule: hairline, tag and uppercase label."""
    st.markdown(f'<div class="lab-sec"><span class="n">{escape(tag)}</span>{escape(label)}</div>', unsafe_allow_html=True)


def section(n: int, title: str = "") -> None:
    rule(f"{n:02d}", SECTIONS[n - 1])
    if title:
        st.subheader(title, anchor=False)


def subsection(code: str, title: str) -> None:
    st.markdown(f"#### {code} {title}")


@contextmanager
def figure(n: str, title: str, subtitle: str = ""):
    """A bordered figure panel: 'Fig. n' label, title and subtitle; the chart and caption go inside."""
    with st.container(border=True):
        sub = f'<div class="sub">{escape(subtitle)}</div>' if subtitle else ""
        st.markdown(f'<div class="fig-head"><span class="num">Fig. {n}</span><span class="title">{escape(title)}'
                    f'</span>{sub}</div>', unsafe_allow_html=True)
        yield


def fig_caption(n: str, text: str) -> None:
    st.caption(f"**Fig. {n} |** {text}")


def table(n: str, title: str, df: pd.DataFrame, note_text: str = "", height: int | None = None) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="fig-head"><span class="num">Table {n}</span><span class="title">{escape(title)}'
                    '</span></div>', unsafe_allow_html=True)
        st.dataframe(df, hide_index=True, **({"height": height} if height else {}))
        if note_text:
            st.caption(note_text)


def note(label: str, text: str) -> None:
    """A labelled interpretation note (rendered as a styled block quote)."""
    st.markdown(f"> **{label}.** " + text.replace("\n", "\n> "))


def table_view(df: pd.DataFrame, label: str = "Data table for this figure") -> None:
    with st.expander(label, icon=":material/table:"):
        st.dataframe(df, hide_index=True)


def subject_record(b: dict, person: dict) -> None:
    meta_grid({"Subject": person["label"], "Description": person["description"],
               "Selected as": f"{person['why_chosen']} (chosen in advance)",
               "Partition": f"{person['split']}: none of the three models was trained on this person",
               "Survey row": person["row_id"], "Measured outcome": f"{num(person['observed'])} {b['units']}"})


def feature_profile(b: dict, values: dict) -> pd.DataFrame:
    """The subject's features against the training distribution (z = (value − mean) / SD)."""
    stats, rows = b["feature_stats"], []
    for f in numeric_features(b):
        s, x = stats[f["name"]], values[f["name"]]
        rows.append({"Feature": f["label"], "Units": f["units"], "Subject": val(x), "Training mean": val(s["train_mean"]),
                     "Training SD": val(s["train_sd"]), "z (SD units)": f"{(x - s['train_mean']) / s['train_sd']:+.2f}"})
    for g in b["categorical_groups"]:
        level = level_of(g, values)
        col = next(lv["column"] for lv in g["levels"] if lv["label"] == level)
        share = stats[col]["train_mean"] if col else 1 - sum(stats[lv["column"]]["train_mean"]
                                                             for lv in g["levels"] if lv["column"])
        rows.append({"Feature": g["label"], "Units": "category", "Subject": level,
                     "Training mean": f"{share:.1%} in this group", "Training SD": "", "z (SD units)": ""})
    return pd.DataFrame(rows)


def research_translation(title: str, lead: str, body: str) -> None:
    section(8, title[:1].upper() + title[1:])
    st.markdown(lead)
    with st.expander(f"Researcher's Perspective: {title}", icon=":material/biotech:"):
        st.markdown(body)
        st.caption("Hypothetical research scenario, for interpretation only. No model was fitted to structural data and "
                   "no results are reported; every number in this module comes from the NHANES models above.")


# ---- Charts ------------------------------------------------------------------
def style(fig: go.Figure, x_title: str, y_title: str, height: int = 340, legend: bool = True) -> go.Figure:
    """Publication-style axes: boxed frame, outside ticks, faint grid, compact legend."""
    c = palette()
    axis = dict(showline=True, linewidth=1, linecolor=c["axis"], mirror=True, ticks="outside", ticklen=4,
                tickcolor=c["axis"], gridcolor=c["grid"], zeroline=False, title_font=dict(size=12),
                tickfont=dict(size=11))
    fig.update_layout(height=height, margin=dict(l=8, r=14, t=34, b=8), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=11),
                                  bgcolor="rgba(0,0,0,0)"),
                      font=dict(family="IBM Plex Sans, sans-serif", size=12), hovermode="closest", bargap=0.45,
                      hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace", size=11)))
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "toImageButtonOptions": {"format": "png", "filename": key, "scale": 3}})


def hline(fig: go.Figure, y: float, name: str, color: str, x0: float, x1: float, width: float = 2) -> None:
    """A labelled horizontal reference line drawn as a trace, so it sits in the legend."""
    fig.add_scatter(x=[x0, x1], y=[y, y], mode="lines", name=name, line=dict(color=color, width=width),
                    hovertemplate=f"{name}<extra></extra>")


def vline(fig: go.Figure, x: float, name: str, color: str, y0, y1, width: float = 2) -> None:
    """A labelled vertical reference line drawn as a trace, so it sits in the legend."""
    fig.add_scatter(x=[x, x], y=[y0, y1], mode="lines", name=name, line=dict(color=color, width=width),
                    hovertemplate=f"{name}<extra></extra>")
