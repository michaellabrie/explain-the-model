"""Push & Pull: a linear prediction rebuilt from a starting point plus one signed push per feature.

push_j = coefficient_j × (value_j − training average_j); starting point + all pushes = prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import m10
from sections.common import (FOCUS_FEATURE, feature, label_of, level_of, load_data, num, numeric_features, outcome,
                             palette, pick_person, push_label, researcher_perspective, short_label, show, signed,
                             step, style, table_view, unit, val, vline)

# Input step and display format for each editable numeric feature.
INPUT = {"age": (1.0, "%.0f"), "hba1c": (0.1, "%.1f"), "bmi": (0.1, "%.1f"), "waist_cm": (0.5, "%.1f"),
         "income_poverty_ratio": (0.05, "%.2f")}
PER = {"years": "year", "ratio": "unit of the ratio"}


def page() -> None:
    b, person = pick_person()
    u, lin = b["units"], b["linear"]
    st.title("Push & Pull")
    st.markdown("##### Which features push a prediction up or down, and by how much?")
    st.markdown(f"A **linear regression** predicts **{outcome(b)} ({u})** by adding up one contribution per "
                f"feature. Here you rebuild its prediction for one real person from a **starting point** plus "
                f"one signed **push** per feature, so you can see where every part of the number comes from.")
    st.latex(r"\hat{y} \;=\; \underbrace{b_0 + \sum_j \beta_j\,\bar{x}_j}_{\text{starting point}}"
             r"\;+\; \sum_j \underbrace{\beta_j\,(x_j - \bar{x}_j)}_{\text{push from feature } j}")
    st.markdown(
        f"This is the usual regression equation $\\hat y = b_0 + \\sum_j \\beta_j x_j$, rearranged so that "
        f"every feature is measured from its training average.\n"
        f"- **Coefficient** $\\beta_j$: how much the prediction changes per one unit of feature $j$, holding the "
        f"other features fixed. One number per feature, **the same for everyone**.\n"
        f"- **Training average** $\\bar x_j$: the mean of feature $j$ over the {b['n_train']} training rows.\n"
        f"- **Push**: $\\beta_j \\times (x_j - \\bar x_j)$. Positive pushes raise the prediction and negative "
        f"pushes lower it. **Different for every person.**\n"
        f"- **Starting point**: the prediction for the training average itself, "
        f"**{num(lin['baseline'])} {u}**. Every person starts here.")

    values, edited = _inputs(b, person)
    ex = m10.linear_explain(b, values)
    observed = None if edited else person["observed"]
    _rebuild(b, person, values, ex, observed)
    _coefficient_vs_push(b, person, values, ex)
    _intercept(b)
    _residuals(b, person, ex, observed)
    researcher_perspective("reading coefficients in a structural-biology model", PERSPECTIVE)


# ---- 1. Inputs ---------------------------------------------------------------
def _reset(keys: list[str]) -> None:
    for k in keys:
        st.session_state.pop(k, None)


def _inputs(b: dict, person: dict) -> tuple[dict, bool]:
    step(1, f"Meet {person['label']}, then try changing their values")
    st.markdown("These are the person's real survey values. Change any of them to watch the pushes respond: "
                "the coefficients, the training averages and the starting point never change.")
    values = dict(person["values"])
    groups, numeric = b["categorical_groups"], numeric_features(b)
    keys = [f"{b['id']}_{person['id']}_{f['name']}" for f in numeric] + \
           [f"{b['id']}_{person['id']}_{g['name']}" for g in groups]
    cols = st.columns(3)
    for i, f in enumerate(numeric):
        s = b["feature_stats"][f["name"]]
        stp, fmt = INPUT.get(f["name"], (0.1, "%.2f"))
        values[f["name"]] = cols[i % 3].number_input(
            f"{f['label']} ({f['units']})", min_value=float(s["min"]), max_value=float(s["max"]),
            value=float(person["values"][f["name"]]), step=stp, format=fmt, key=keys[i])
    for j, g in enumerate(groups):
        levels = [lv["label"] for lv in g["levels"]]
        choice = cols[(len(numeric) + j) % 3].selectbox(g["label"], levels,
                                                        index=levels.index(level_of(g, person["values"])),
                                                        key=keys[len(numeric) + j])
        values.update({lv["column"]: float(lv["label"] == choice) for lv in g["levels"] if lv["column"]})

    edited = any(values[k] != person["values"][k] for k in values)
    st.button("Reset to the supplied values", on_click=_reset, args=(keys,), disabled=not edited,
              icon=":material/restart_alt:")
    if edited:
        st.warning(f"These are edited values, so this is no longer {person['label']}'s real data. Their measured "
                   "outcome no longer applies and is hidden below; the model's prediction is computed live.",
                   icon=":material/edit:")
    return values, edited


# ---- 2. Rebuild the prediction -------------------------------------------------
def _push_parts(b: dict, values: dict, key: str) -> dict:
    """How one grouped push is computed, as display strings."""
    lin = b["linear"]
    g = next((g for g in b["categorical_groups"] if g["name"] == key), None)
    if g is None:
        f = feature(b, key)
        coef, ref, x = lin["coefficients"][key], lin["reference"][key], values[key]
        return dict(value=f"{val(x)}{unit(f['units'])}", ref=f"{val(ref)}{unit(f['units'])}",
                    coef=f"{val(coef)} per {PER.get(f['units'], f['units'])}", diff=signed(x - ref, 2),
                    how=f"{val(coef)} × ({val(x)} − {val(ref)})")
    cols = [lv for lv in g["levels"] if lv["column"]]
    if len(cols) == 1:
        c = cols[0]["column"]
        coef, ref = lin["coefficients"][c], lin["reference"][c]
        return dict(value=level_of(g, values), ref=f"{ref:.0%} {cols[0]['label']}",
                    coef=f"{val(coef)} ({cols[0]['label']} vs {g['reference']})",
                    diff=signed(values[c] - ref, 2), how=f"{val(coef)} × ({values[c]:g} − {ref:.3f})")
    return dict(value=level_of(g, values), ref="training mix", coef=f"one per group, vs {g['reference']}",
                diff="see below", how=f"the sum of {len(cols)} terms, one per 0/1 column")


def _rebuild(b: dict, person: dict, values: dict, ex: dict, observed: float | None) -> None:
    u = b["units"]
    pushes = ex["grouped_pushes"]
    step(2, "Rebuild the prediction: starting point + pushes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Starting point ({u})", num(ex["baseline"]), help="The prediction for the training average.")
    c2.metric(f"Sum of pushes ({u})", signed(sum(pushes.values())))
    c3.metric(f"Prediction ({u})", num(ex["prediction"]), help="Starting point + sum of pushes.")
    c4.metric(f"Measured ({u})", num(observed) if observed is not None else "hidden",
              help="The person's real outcome (hidden when the values are edited).")

    ups = {k: v for k, v in pushes.items() if v > 0}
    downs = {k: v for k, v in pushes.items() if v < 0}
    up = max(ups, key=ups.get) if ups else None
    down = min(downs, key=downs.get) if downs else None
    st.markdown(
        (f"Biggest upward push: **{push_label(b, up)} ({signed(ups[up])} {u})**. " if up else
         "No feature pushes this prediction upward. ") +
        (f"Biggest downward push: **{push_label(b, down)} ({signed(downs[down])} {u})**." if down else
         "No feature pushes this prediction downward."))
    _waterfall(b, values, ex, observed)

    terms = " ".join(f"{'+' if v >= 0 else '−'} {abs(v):.1f}" for v in pushes.values())
    st.markdown(f"In numbers: **{num(ex['baseline'])} {terms} = {num(ex['prediction'])} {u}** "
                "(each push rounded to one decimal).")
    rows = []
    for k, v in pushes.items():
        p = _push_parts(b, values, k)
        rows.append({"Feature": push_label(b, k), "This person": p["value"], "Training average": p["ref"],
                     "Value − average": p["diff"], f"Coefficient ({u})": p["coef"], f"Push ({u})": signed(v, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Each push = coefficient × (value − training average). The table's pushes, added to the starting "
               "point, give the prediction exactly. These are associations in the survey data, not causes: "
               "the model says what it predicts for people like this, not what would happen if a value changed.")
    _category_breakdown(b, values, ex)

    with st.expander("Check yourself: rebuild it by hand", icon=":material/calculate:"):
        st.markdown(f"Using only the starting point ({num(ex['baseline'])} {u}) and the pushes in the table, "
                    "what does the model predict?")
        guess = st.number_input(f"Your prediction ({u})", value=None, step=0.1, format="%.1f",
                                key=f"guess_{b['id']}_{person['id']}")
        if guess is not None:
            if abs(guess - ex["prediction"]) <= 0.15 + 0.05 * len(pushes):
                st.success(f"Yes: {num(ex['baseline'])} plus all the pushes is {num(ex['prediction'])} {u}.")
            else:
                st.error("Not quite. Start from the starting point (not the intercept) and add every push "
                         "with its sign.")


def _waterfall(b: dict, values: dict, ex: dict, observed: float | None) -> None:
    c, u = palette(), b["units"]
    items = sorted(ex["grouped_pushes"].items(), key=lambda kv: -abs(kv[1]))
    labels = [push_label(b, k) for k, _ in items]
    starts = list(np.cumsum([ex["baseline"]] + [v for _, v in items])[:-1])
    ends = [s + v for s, (_, v) in zip(starts, items)]
    rows = ["Starting point", *labels, "Prediction"]

    fig = go.Figure()
    cx, cy = [], []
    for i, x in enumerate([ex["baseline"], *ends]):
        cx += [x, x, None]
        cy += [rows[i], rows[i + 1], None]
    fig.add_scatter(x=cx, y=cy, mode="lines", line=dict(color=c["faint"], width=1), hoverinfo="skip",
                    showlegend=False)
    hover = [f"<b>{lab}</b><br>push = {_push_parts(b, values, k)['how']}<br>= {signed(v, 2)} {u}"
             f"<br>running total {num(s)} → {num(s + v)} {u}" for lab, (k, v), s in zip(labels, items, starts)]
    fig.add_bar(x=[v for _, v in items], y=labels, base=starts, orientation="h", width=0.5,
                marker=dict(color=[c["red"] if v > 0 else c["blue"] for _, v in items], cornerradius=4),
                text=[signed(v) for _, v in items], textposition="outside", cliponaxis=False,
                hovertext=hover, hoverinfo="text", showlegend=False)
    fig.add_scatter(x=[ex["baseline"]], y=["Starting point"], mode="markers+text", showlegend=False,
                    text=[f"  {num(ex['baseline'])} {u}"], textposition="middle right",
                    marker=dict(symbol="diamond", size=13, color=c["muted"]),
                    hovertemplate=f"Starting point: prediction for the training average<br>{num(ex['baseline'])} {u}<extra></extra>")
    fig.add_scatter(x=[ex["prediction"]], y=["Prediction"], mode="markers+text", showlegend=False,
                    text=[f"  {num(ex['prediction'])} {u}"], textposition="middle right",
                    marker=dict(size=13, color=c["blue"], line=dict(width=2, color=c["surface"])),
                    hovertemplate=f"Prediction = starting point + all pushes<br>{num(ex['prediction'])} {u}<extra></extra>")
    pts = [ex["baseline"], ex["prediction"], *ends]
    if observed is not None:
        pts.append(observed)
        fig.add_vline(x=observed, line_color=c["aqua"], line_width=2,
                      annotation_text=f"Measured: {num(observed)} {u}", annotation_position="top")
    lo, hi = min(pts), max(pts)
    pad = 0.2 * (hi - lo or 1.0)
    style(fig, f"{outcome(b)} ({u})", "", height=110 + 46 * len(rows), legend=False)
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(categoryorder="array", categoryarray=rows, autorange="reversed")
    fig.update_layout(margin=dict(t=48))
    show(fig, key="linear_waterfall")
    st.caption("Read from top to bottom: start at the diamond, slide right for each red (upward) push and left for "
               "each blue (downward) push, and you land on the prediction. Largest pushes are listed first; the "
               "order doesn't change the total.")


def _category_breakdown(b: dict, values: dict, ex: dict) -> None:
    lin, u = b["linear"], b["units"]
    with st.expander("How a category (sex, ethnicity) becomes one push", icon=":material/category:"):
        st.markdown("The model stores each category as 0/1 columns, one per group except a **reference group** "
                    "that is all zeros. Each column gets its own term, coefficient × (0 or 1 − training share), "
                    "and the terms are added into one push. Because the training share is an average mix, even a "
                    "person in the reference group gets a small push. That's correct, not a bug.")
        for g in b["categorical_groups"]:
            cols = [lv for lv in g["levels"] if lv["column"]]
            rows = [{"Column (group)": lv["label"], f"Coefficient vs {g['reference']} ({u})": val(lin["coefficients"][lv["column"]]),
                     "Training share": f"{lin['reference'][lv['column']]:.3f}", "This person": f"{values[lv['column']]:g}",
                     f"Term ({u})": signed(ex["pushes"][lv["column"]], 2)} for lv in cols]
            st.markdown(f"**{g['label']}** (reference group: {g['reference']}): "
                        f"push = **{signed(ex['grouped_pushes'][g['name']], 2)} {u}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---- 3. Coefficient vs push ----------------------------------------------------
def _coefficient_vs_push(b: dict, person: dict, values: dict, ex: dict) -> None:
    c, u, lin = palette(), b["units"], b["linear"]
    step(3, "A coefficient is a slope; a push is where one person sits on it")
    numeric = [f["name"] for f in numeric_features(b)]
    default = max(numeric, key=lambda n: abs(ex["pushes"][n]))
    name = st.selectbox("Feature to inspect", numeric, index=numeric.index(default),
                        format_func=lambda n: label_of(b, n), key=f"slope_{b['id']}_{person['id']}")
    f, s = feature(b, name), b["feature_stats"][name]
    coef, ref, x, push = lin["coefficients"][name], lin["reference"][name], values[name], ex["pushes"][name]
    per = PER.get(f["units"], f["units"])

    grid = np.linspace(min(s["p01"], x), max(s["p99"], x), 60)
    line = coef * (grid - ref)
    others = [p for p in b["people"] if p["id"] != person["id"]]
    fig = go.Figure()
    fig.add_hline(y=0, line_color=c["faint"], line_width=1)
    vline(fig, ref, f"Training average: {val(ref)}{unit(f['units'])} (push = 0)", c["muted"], line.min(), line.max(), 1.5)
    fig.add_scatter(x=grid, y=line, mode="lines", name=f"Push for any value: slope = coefficient = {val(coef)} {u} per {per}",
                    line=dict(color=c["blue"], width=2),
                    hovertemplate=f"value %{{x:.3g}}<br>push %{{y:+.2f}} {u}<extra></extra>")
    fig.add_scatter(x=[p["values"][name] for p in others], y=[lin["people"][p["id"]]["pushes"][name] for p in others],
                    mode="markers+text", name="Other example people (supplied values)",
                    text=[p["label"].replace("Example person ", "P") for p in others], textposition="top center",
                    marker=dict(size=9, color=c["muted"], line=dict(width=2, color=c["surface"])),
                    hovertemplate=f"value %{{x:.3g}}<br>push %{{y:+.2f}} {u}<extra></extra>")
    fig.add_scatter(x=[x], y=[push], mode="markers", name=f"{person['label']}: push {signed(push)} {u}",
                    marker=dict(size=14, color=c["orange"], line=dict(width=2, color=c["surface"])),
                    hovertemplate=f"value %{{x:.3g}}<br>push %{{y:+.2f}} {u}<extra></extra>")
    show(style(fig, f"{f['label']} ({f['units']})", f"Push ({u})", height=400), key="linear_slope")

    st.latex(rf"\text{{push}} = \beta \times (x - \bar x) = {val(coef)} \times ({val(x)} - {val(ref)}) "
             rf"= {signed(push, 2)}\ \text{{{u}}}")
    side = "above" if x > ref else "below" if x < ref else "exactly at"
    st.markdown(f"The **coefficient** ({val(coef)} {u} per {per}) is the slope of the line. It is the same for all "
                f"five example people, and for everyone else. The **push** depends on where one person sits: "
                f"{person['label']}'s value is {side} the training average, so the push is **{signed(push)} {u}**. "
                "Same slope, different positions, different pushes.")

    for g in b["categorical_groups"]:
        _category_text(b, g, values, ex)
    _general_vs_person(b, person, ex)


def _category_text(b: dict, g: dict, values: dict, ex: dict) -> None:
    lin, u = b["linear"], b["units"]
    cols = [lv for lv in g["levels"] if lv["column"]]
    push = ex["grouped_pushes"][g["name"]]
    level = level_of(g, values)
    st.markdown(f"**{g['label']}: two different comparisons.**")
    if len(cols) == 1:
        lv = cols[0]
        coef, share = lin["coefficients"][lv["column"]], lin["reference"][lv["column"]]
        st.markdown(f"- **Coefficient:** {val(coef)} {u}. Holding the other features fixed, the model predicts "
                    f"{num(abs(coef))} {u} {'lower' if coef < 0 else 'higher'} for *{lv['label']}* than for the "
                    f"**reference group**, *{g['reference']}*. That's just how the 0/1 column is coded.\n"
                    f"- **This person's push:** {signed(push)} {u} for *{level}*, measured from the **training mix** "
                    f"({share:.0%} {lv['label']}), not from the reference group.")
    else:
        rows = [{"Group": lv["label"], f"Coefficient vs {g['reference']} ({u})": val(lin["coefficients"][lv["column"]]),
                 "Share of training rows": f"{lin['reference'][lv['column']]:.1%}"} for lv in cols]
        st.markdown(f"- **Coefficients** compare each group with the **reference group**, *{g['reference']}*, "
                    f"holding the other features fixed. For example, *{cols[0]['label']}*: "
                    f"{signed(lin['coefficients'][cols[0]['column']])} {u} vs *{g['reference']}*.\n"
                    f"- **This person's push:** {signed(push)} {u} for *{level}*, measured from the **training mix** "
                    "of all groups.")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def _general_vs_person(b: dict, person: dict, ex: dict) -> None:
    c, u, lin = palette(), b["units"], b["linear"]
    st.markdown("**In general vs. for this person.** Raw coefficients can't be compared across features, because "
                "each is *per one unit* of a different thing (per %, per year, per cm). Multiplying a coefficient by "
                "the feature's training standard deviation (SD) gives the change in prediction for a *typical* "
                "difference between people. That is a fair, general comparison. A push is one person's story: "
                "a feature can matter a lot in general and still give a small push to someone near the average.")
    names = sorted(lin["typical_effect_per_sd"], key=lambda n: -abs(lin["typical_effect_per_sd"][n]))
    labels = [short_label(b, n) for n in names]
    fig = go.Figure()
    fig.add_bar(y=labels, x=[lin["typical_effect_per_sd"][n] for n in names], orientation="h",
                name="In general: effect of a 1-SD difference (coefficient × SD)", marker=dict(color=c["blue"], cornerradius=4),
                hovertemplate=f"%{{y}}<br>1-SD effect %{{x:+.2f}} {u}<extra></extra>")
    fig.add_bar(y=labels, x=[ex["pushes"][n] for n in names], orientation="h",
                name=f"{person['label']}: push", marker=dict(color=c["orange"], cornerradius=4),
                hovertemplate=f"%{{y}}<br>this person's push %{{x:+.2f}} {u}<extra></extra>")
    fig.add_vline(x=0, line_color=c["muted"], line_width=1)
    style(fig, f"Change in predicted {outcome(b)} ({u})", "", height=120 + 60 * len(names))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig.update_yaxes(autorange="reversed")
    show(fig, key="linear_general")
    table_view(pd.DataFrame([{
        "Feature": label_of(b, n), f"Coefficient ({u} per unit)": val(lin["coefficients"][n]),
        "Training SD": val(b["feature_stats"][n]["train_sd"]),
        f"Typical effect per SD ({u})": signed(lin["typical_effect_per_sd"][n], 2),
        f"This person's push ({u})": signed(ex["pushes"][n], 2)} for n in names]))
    st.caption("Sex and ethnicity aren't on this chart: a 0/1 category has no 'typical 1-SD difference'. Their "
               "coefficients are group comparisons (above).")


# ---- 4. Intercept --------------------------------------------------------------
def _intercept(b: dict) -> None:
    lin, u = b["linear"], b["units"]
    step(4, "Why the intercept is not the starting point")
    terms = {f: lin["coefficients"][f] * lin["reference"][f] for f in lin["coefficients"]}
    zero = [f"{short_label(b, f['name'])} 0{unit(f['units'])}" for f in numeric_features(b)]
    zero += [g["reference"].lower() if g["name"] == "sex" else g["reference"] for g in b["categorical_groups"]]
    st.markdown(f"The **intercept** $b_0$ = **{num(lin['intercept'], 2)} {u}** is the prediction when *every* "
                f"feature equals 0: {', '.join(zero)}. No such person exists, so the intercept is "
                "just where the fitted line would cross zero, far outside the data. The **starting point** moves "
                "from that impossible person to the training average:")
    st.latex(rf"\text{{starting point}} = b_0 + \sum_j \beta_j \bar x_j = {num(lin['intercept'], 2)} "
             rf"{'+' if sum(terms.values()) >= 0 else '-'} {num(abs(sum(terms.values())), 2)} "
             rf"= {num(lin['baseline'], 2)}\ \text{{{u}}}")
    table_view(pd.DataFrame([{"Feature": label_of(b, f), "Coefficient β": val(lin["coefficients"][f]),
                              "Training average x̄": val(lin["reference"][f]), f"β × x̄ ({u})": num(t, 2)}
                             for f, t in terms.items()]), "Show each β × x̄ term")
    st.caption("Same model, same predictions: only the reference point differs. Pushes measured from the "
               "training average stay small and readable; 'pushes' measured from zero would be huge and meaningless.")


# ---- 5. Residuals --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _validation_rows(key: str) -> pd.DataFrame:
    """Validation rows with the fixed linear model's prediction (via m10.linear_explain) and residual."""
    b = m10.load_bundle(key)
    df = load_data(key)
    rows = df[df.split == "validation"].copy()
    rows["prediction"] = [m10.linear_explain(b, r)["prediction"]
                          for r in rows[m10.feature_names(b)].to_dict("records")]
    rows["residual"] = rows[b["target"]] - rows["prediction"]
    return rows


def _residuals(b: dict, person: dict, ex: dict, observed: float | None) -> None:
    c, u, lin = palette(), b["units"], b["linear"]
    step(5, "Residuals: where the straight line fits, and where it fails")
    rmse = lin["rmse"]
    st.markdown(f"A **residual** is measured − predicted. On the {b['n_validation']} validation people (never used "
                f"for fitting), the typical residual size, the **root-mean-square error (RMSE)**, is "
                f"**{num(rmse['validation'])} {u}** (training rows: {num(rmse['train'])} {u}).")
    if observed is not None:
        res = observed - ex["prediction"]
        st.markdown(f"For {person['label']}: {num(observed)} − {num(ex['prediction'])} = **{signed(res)} {u}**, "
                    f"about {abs(res) / rmse['validation']:.1f}× the typical error.")
        if abs(res) > 1.5 * rmse["validation"]:
            st.warning(f"The line misses {person['label']} by a lot: it predicts "
                       f"{ex['prediction'] / observed:.1f}× the measured value. That is the model, not a bug. A single "
                       "straight line has one slope per feature for everyone, so it can't follow a relationship "
                       "that bends. The residual plot below shows where that happens.", icon=":material/info:")

    rows = _validation_rows(b["id"])
    numeric = [f["name"] for f in numeric_features(b)]
    name = st.selectbox("Plot against", numeric, index=numeric.index(FOCUS_FEATURE.get(b["id"], numeric[0])),
                        format_func=lambda n: label_of(b, n), key=f"resid_{b['id']}")
    f = feature(b, name)
    me = rows[rows.row_id == person["row_id"]] if observed is not None else rows.iloc[0:0]
    tab1, tab2 = st.tabs(["Measured vs. feature, with the model's line", "Residuals vs. feature"])
    with tab1:
        grid = np.linspace(rows[name].min(), rows[name].max(), 60)
        line = [m10.linear_explain(b, {**lin["reference"], name: g})["prediction"] for g in grid]
        fig = go.Figure()
        fig.add_scatter(x=rows[name], y=rows[b["target"]], mode="markers", name="Validation people (measured)",
                        marker=dict(size=6, color=c["muted"], opacity=0.45),
                        hovertemplate=f"{f['label']} %{{x}}<br>measured %{{y:.1f}} {u}<extra></extra>")
        fig.add_scatter(x=grid, y=line, mode="lines", name="Model's prediction (other features at training average)",
                        line=dict(color=c["blue"], width=2.5))
        if len(me):
            fig.add_scatter(x=me[name], y=me[b["target"]], mode="markers", name=person["label"],
                            marker=dict(size=14, color=c["orange"], line=dict(width=2, color=c["surface"])))
        show(style(fig, f"{f['label']} ({f['units']})", f"Measured {outcome(b)} ({u})", height=420), key="linear_fit")
    with tab2:
        bins = pd.qcut(rows[name], 10, duplicates="drop")
        binned = rows.groupby(bins, observed=True)[[name, "residual"]].mean()
        fig = go.Figure()
        fig.add_hline(y=0, line_color=c["muted"], line_width=1)
        fig.add_scatter(x=rows[name], y=rows["residual"], mode="markers", name="Validation people",
                        marker=dict(size=6, color=c["muted"], opacity=0.45),
                        hovertemplate=f"{f['label']} %{{x}}<br>residual %{{y:+.1f}} {u}<extra></extra>")
        fig.add_scatter(x=binned[name], y=binned["residual"], mode="lines+markers",
                        name="Average residual (10 equal-size groups)", line=dict(color=c["violet"], width=2.5),
                        marker=dict(size=8, line=dict(width=2, color=c["surface"])))
        if len(me):
            fig.add_scatter(x=me[name], y=me["residual"], mode="markers", name=person["label"],
                            marker=dict(size=14, color=c["orange"], line=dict(width=2, color=c["surface"])))
        show(style(fig, f"{f['label']} ({f['units']})", f"Residual: measured − predicted ({u})", height=420),
             key="linear_resid")
    st.markdown("**How to read it.** If a straight line is a good description, residuals scatter evenly around zero "
                "at every value of the feature and the violet average stays flat. A curve or a slope in the "
                "violet line means the model's errors depend on that feature in a way one straight line can't "
                "capture. It is a pattern to explain, not noise.")
    if b["id"] == "fev1":
        st.info("Lung function shows this clearly against age: lungs grow through childhood and shrink later in "
                "life, but the model has one slope for age. So it over-predicts for children (residuals below "
                "zero), under-predicts for young adults and over-predicts again for older adults. Height, which "
                "would explain much of this, was deliberately left out of the data.", icon=":material/info:")


PERSPECTIVE = """
**A hypothetical question.** Could structural features measured on capsid-subunit interfaces predict a continuous
biochemical property such as the **interface binding free energy** (ΔG, kcal/mol)? Picture a table with one row per
protein–protein interface, for example the 2-fold, 3-fold and 5-fold symmetry interfaces of AAV capsid variants
solved by cryo-EM. The columns would be structure-derived features: buried surface area (Å²), hydrogen bonds, salt
bridges, shape complementarity, local map resolution (Å). A linear model of that table would work exactly like
this page:

- **Starting point:** the predicted ΔG for an interface with every feature at its training average, a reference
  interface rather than a real one.
- **Coefficient** on buried area: the change in predicted ΔG per extra Å², *holding the other features fixed*. One
  slope, shared by every interface.
- **Push:** coefficient × (this interface's buried area − the average buried area). The pushes add up to that
  interface's prediction, so you could say which features make it predicted to bind more or less tightly than
  average, and by how many kcal/mol.
- **Intercept:** the prediction for an interface with zero buried area and zero contacts. As here, it's physically
  meaningless.
- **Categories:** an "interface type" (2-fold / 3-fold / 5-fold) would be coded against a reference level, much as
  mutants are reported against wild type. The coefficient compares with that reference; a push compares with the
  training mix.
- **Residuals:** if large interfaces were systematically under-predicted, the residual plot would curve, the same
  lesson FEV₁ against age teaches. The fix is a better model (a non-linear term, or the tree models in the other
  sections), not a hidden plot.

**Interpretation discipline.** Coefficients describe *associations across structures*, not mechanisms. Buried area
and hydrogen-bond count rise together, so "holding the others fixed" can describe interfaces that never occur, and
correlated features share credit unpredictably. A positive coefficient doesn't show that engineering more buried
area would strengthen binding; that needs a designed experiment, such as mutagenesis with measured affinities.
Features read off cryo-EM maps also carry measurement error that varies with local resolution, and noisy features
tend to bias their coefficients toward zero (*regression dilution*).
"""
