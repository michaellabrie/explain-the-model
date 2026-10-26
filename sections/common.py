"""Shared pieces for every section: the dataset and person picker, number formats,
chart styling and the collapsible Researcher's Perspective box.

Every number shown anywhere in the app comes from outputs/*.json or the m10 helpers.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import m10

LABELS = {"glucose": "Fasting glucose", "sbp": "Blood pressure", "fev1": "Lung function (FEV₁)"}
# Feature that residual plots show first. For lung function, age shows where one straight line fails.
FOCUS_FEATURE = {"glucose": "hba1c", "sbp": "age", "fev1": "age"}

# One categorical palette with separate light and dark steps (checked for colour-blind separation).
# Roles used across the app: blue = model prediction / training data, orange = validation data or
# the highlighted item, aqua = a measured (observed) value, red/blue = upward/downward push,
# violet = the learning-rate-0.02 comparison model, ink = a "you are here" marker.
_LIGHT = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#6250d6", red="#e34948",
              muted="#898781", faint="#d3d2cc", ink="#0b0b0b", surface="#ffffff")
_DARK = dict(blue="#3987e5", orange="#d95926", aqua="#199e70", violet="#9085e9", red="#e66767",
             muted="#898781", faint="#4d4c48", ink="#f0efec", surface="#0e1117")


def palette() -> dict:
    try:
        return _DARK if st.context.theme.type == "dark" else _LIGHT
    except Exception:  # no theme information (e.g. headless tests)
        return _LIGHT


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


def outcome(b: dict) -> str:
    """Short name of the predicted outcome, e.g. 'Fasting glucose' or 'FEV₁'."""
    return b["target_label"].split(" (")[0].split(":")[0]


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


# ---- Layout ------------------------------------------------------------------
def pick_person() -> tuple[dict, dict]:
    """Sidebar dataset and example-person pickers. Every section starts here."""
    st.sidebar.markdown("### :material/biotech: The Researcher's Regression Lab")
    key = st.sidebar.selectbox("Dataset", m10.DATASETS, format_func=LABELS.get, key="dataset")
    b = m10.load_bundle(key)
    labels = {p["id"]: f"{p['label']}: {p['description']}" for p in b["people"]}
    pid = st.sidebar.selectbox("Example person", list(labels), format_func=labels.get, key=f"person_{key}")
    person = next(p for p in b["people"] if p["id"] == pid)

    with st.sidebar:
        st.markdown(f"**{person['label']}** ({person['description']}) was picked as a *{person['why_chosen']}*. "
                    f"Measured {outcome(b)}: **{num(person['observed'])} {b['units']}**.")
        st.caption(f"A real survey participant from the {person['split']} rows: none of the three models "
                   "was trained on this person. The same five people appear in every section.")
        with st.expander("About this dataset", icon=":material/dataset:"):
            feats = "".join(f"\n- {f['label']} ({f['units']})" for f in numeric_features(b))
            cats = "".join(f"\n- {g['label']}: {', '.join(lv['label'] for lv in g['levels'])}"
                           for g in b["categorical_groups"])
            st.markdown(f"**{b['name']}**  \nPredicts **{b['target_label']}** in **{b['units']}** from:{feats}{cats}")
            st.caption(f"{b['n_train']} training rows and {b['n_validation']} validation rows "
                       f"({b['split_description']}). Source: [{b['source']}]({b['url']}). "
                       "These describe this survey sample, not the whole US population.")
    return b, person


def step(n: int, title: str) -> None:
    st.subheader(f"{n}. {title}", divider="gray")


def researcher_perspective(title: str, body: str) -> None:
    with st.expander(f"Researcher's Perspective: {title}", icon=":material/biotech:"):
        st.markdown(body)
        st.caption("A hypothetical research scenario, for intuition only. No model was fitted to it and no results "
                   "are reported; every number on this page comes from the NHANES models above.")


def table_view(df: pd.DataFrame, label: str = "Show the numbers behind this chart") -> None:
    with st.expander(label, icon=":material/table:"):
        st.dataframe(df, hide_index=True)


# ---- Charts ------------------------------------------------------------------
def style(fig: go.Figure, x_title: str, y_title: str, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      hovermode="closest", bargap=0.45)
    fig.update_xaxes(title_text=x_title, zeroline=False)
    fig.update_yaxes(title_text=y_title, zeroline=False)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def hline(fig: go.Figure, y: float, name: str, color: str, x0: float, x1: float, width: float = 2) -> None:
    """A labelled horizontal reference line drawn as a trace, so it sits in the legend."""
    fig.add_scatter(x=[x0, x1], y=[y, y], mode="lines", name=name, line=dict(color=color, width=width),
                    hovertemplate=f"{name}<extra></extra>")


def vline(fig: go.Figure, x: float, name: str, color: str, y0, y1, width: float = 2) -> None:
    """A labelled vertical reference line drawn as a trace, so it sits in the legend."""
    fig.add_scatter(x=[x, x], y=[y0, y1], mode="lines", name=name, line=dict(color=color, width=width),
                    hovertemplate=f"{name}<extra></extra>")
