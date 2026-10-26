"""Module 1 (Push & Pull): a linear prediction rebuilt from a starting point plus one signed push per feature.

push_j = coefficient_j × (value_j − training average_j); starting point + all pushes = prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import m10
from sections.common import (FOCUS_FEATURE, feature, feature_profile, fig_caption, figure, label_of, level_of,
                             load_data, note, num, numeric_features, outcome, page_header, palette, pick_person,
                             push_label, research_translation, section, short_label, show, signed, style,
                             subject_record, subsection, table, table_view, unit, val, vline)

# Input step and display format for each editable numeric feature.
INPUT = {"age": (1.0, "%.0f"), "hba1c": (0.1, "%.1f"), "bmi": (0.1, "%.1f"), "waist_cm": (0.5, "%.1f"),
         "income_poverty_ratio": (0.05, "%.2f")}
PER = {"years": "year", "ratio": "unit of the ratio"}


def page() -> None:
    b, person = pick_person()
    u = b["units"]
    page_header(b, 1, "Push & Pull", "Linear regression", "Additive Decomposition of a Linear Regression Prediction",
                "Which features push a prediction up or down, and by how much?",
                f"Ordinary least-squares linear regression · {len(b['features'])} predictor columns · "
                "fitted on the training rows")

    section(1)
    st.markdown(
        f"**Question.** For one held-out survey participant, how does the fitted linear model arrive at its "
        f"prediction of {outcome(b, True)} ({u}), and which features move that prediction up or down?\n\n"
        f"**Learning objective.** Rebuild a prediction from a starting point and one signed push per feature, and "
        f"tell a *coefficient* (a property of the model, shared by everyone) apart from one person's *push* (a "
        f"property of that person).\n\n"
        f"**Approach.** Re-express the regression equation around the training averages, so that each feature "
        f"contributes one signed, additive term that can be read, plotted and summed.")

    section(2, "From the regression equation to signed contributions")
    _workflow(b)

    section(3)
    subject_record(b, person)

    section(4, "Feature vector")
    values, edited = _inputs(b, person)
    ex = m10.linear_explain(b, values)
    observed = None if edited else person["observed"]

    section(5, "Reconstruction of the prediction")
    _rebuild(b, person, values, ex, observed)

    section(6, "Coefficients, contributions and the reference point")
    _coefficient_vs_push(b, person, values, ex)
    _general_vs_person(b, person, ex)
    _intercept(b)

    section(7, "Residual structure and the limits of a straight line")
    _residuals(b, person, ex, observed)
    subsection("7.2", "Associations, not mechanisms")
    st.markdown("Every coefficient here summarizes how the outcome *co-varies* with a feature across this survey "
                "sample, holding the other features fixed. It does not say what would happen to a person if a value "
                "changed. Read a push as *the model predicts this much higher or lower for people like this*, never "
                "as *changing this feature would change the outcome*.")
    research_translation("additive models of interface energetics",
                         "How the decomposition above would carry over to a structure-based regression problem in "
                         "capsid biology, and which features of real structural data would limit its interpretation.",
                         PERSPECTIVE)


# ---- 02. Workflow ------------------------------------------------------------
def _workflow(b: dict) -> None:
    u, lin = b["units"], b["linear"]
    st.latex(r"\hat{y} \;=\; \underbrace{b_0 + \sum_j \beta_j\,\bar{x}_j}_{\text{starting point}}"
             r"\;+\; \sum_j \underbrace{\beta_j\,(x_j - \bar{x}_j)}_{\text{push from feature } j}")
    st.markdown("This is the usual regression equation $\\hat y = b_0 + \\sum_j \\beta_j x_j$, rearranged so that "
                "every feature is measured from its training average. The analysis proceeds in six steps:\n"
                "1. **Reference point.** Evaluate the model at the training average of every feature: the "
                "*starting point*.\n"
                "2. **Centering.** Measure each of the subject's features from its training average, "
                "$x_j - \\bar x_j$.\n"
                "3. **Contribution.** Scale by the coefficient: $\\text{push}_j = \\beta_j (x_j - \\bar x_j)$. Sex and "
                "ethnicity are combined into one push each.\n"
                "4. **Reconstruction.** Add the pushes to the starting point; this recovers the prediction exactly.\n"
                "5. **Validation.** Compare with the measured value; the difference is the residual.\n"
                "6. **Diagnosis.** Look for structure in the residuals across all validation participants.")
    st.markdown(
        "| Symbol | Meaning | In this dataset |\n|---|---|---|\n"
        f"| $\\hat y$ | Predicted {outcome(b, True)} | {u} |\n"
        "| $x_j$ | The subject's value of feature $j$ | §04 |\n"
        f"| $\\bar x_j$ | Training average of feature $j$ | mean of {b['n_train']} training rows |\n"
        "| $\\beta_j$ | **Coefficient**: change in prediction per 1 unit of $x_j$, others held fixed. The same for "
        "everyone | Table 2 |\n"
        "| $\\beta_j(x_j-\\bar x_j)$ | **Push**: signed contribution of feature $j$ for this subject. Different for "
        "everyone | Fig. 1 |\n"
        f"| $b_0+\\sum_j\\beta_j\\bar x_j$ | **Starting point**: the prediction at the training average | "
        f"{num(lin['baseline'])} {u} |\n"
        f"| $b_0$ | **Intercept**: the prediction with every feature at 0. Not the starting point | "
        f"{num(lin['intercept'], 2)} {u} |")


# ---- 04. Feature vector --------------------------------------------------------
def _reset(keys: list[str]) -> None:
    for k in keys:
        st.session_state.pop(k, None)


def _inputs(b: dict, person: dict) -> tuple[dict, bool]:
    st.markdown("The subject's recorded survey values. Edit any of them to test how each push responds; the "
                "coefficients, training averages and starting point are fixed properties of the fitted model and do "
                "not change.")
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
    st.button("Restore the recorded values", on_click=_reset, args=(keys,), disabled=not edited,
              icon=":material/restart_alt:")
    if edited:
        note("State", f"These are edited values, so this is no longer {person['label']}'s real data. Their measured "
                      "outcome no longer applies and is hidden below; the model's prediction is computed live.")
    table("1", "Standardized feature profile", feature_profile(b, values),
          "z = (subject value − training mean) / training SD: how many standard deviations the subject sits from "
          "the average training participant. Category rows give the share of training rows in the subject's group.")
    return values, edited


# ---- 05. Reconstruction --------------------------------------------------------
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

    with figure("1", f"Additive decomposition of the predicted {outcome(b, True)}",
                f"{person['label']} · signed contribution of each feature, measured from the training average ({u})"):
        _waterfall(b, values, ex, observed)
        fig_caption("1", "Read from top to bottom. The diamond marks the starting point, the model's prediction at the "
                         "training average. Each bar is one push: red bars raise the prediction, blue bars lower it, "
                         "and each begins where the previous one ended. The final dot is the prediction; the vertical "
                         "line is the measured value. Pushes are ordered by size, and the order does not change the "
                         "total.")

    terms = " ".join(f"{'+' if v >= 0 else '−'} {abs(v):.1f}" for v in pushes.values())
    st.markdown(f"In numbers: **{num(ex['baseline'])} {terms} = {num(ex['prediction'])} {u}** "
                "(each push rounded to one decimal).")
    rows = []
    for k, v in pushes.items():
        p = _push_parts(b, values, k)
        rows.append({"Feature": push_label(b, k), "Subject": p["value"], "Training mean": p["ref"],
                     "Value − mean": p["diff"], f"Coefficient ({u})": p["coef"], f"Push ({u})": signed(v, 2)})
    table("2", "Contribution ledger", pd.DataFrame(rows),
          "Each push = coefficient × (value − training average). Added to the starting point, the pushes in this "
          "table give the prediction exactly.")
    _category_breakdown(b, values, ex)

    with st.expander("Check yourself: rebuild it by hand", icon=":material/calculate:"):
        st.markdown(f"Using only the starting point ({num(ex['baseline'])} {u}) and the pushes in Table 2, "
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
                marker=dict(color=[c["red"] if v > 0 else c["blue"] for _, v in items], cornerradius=2),
                text=[signed(v) for _, v in items], textposition="outside", cliponaxis=False,
                hovertext=hover, hoverinfo="text", showlegend=False)
    fig.add_scatter(x=[ex["baseline"]], y=["Starting point"], mode="markers+text", showlegend=False, cliponaxis=False,
                    text=[f"  {num(ex['baseline'])} {u}"], textposition="middle right",
                    marker=dict(symbol="diamond", size=12, color=c["muted"]),
                    hovertemplate=f"Starting point: prediction for the training average<br>{num(ex['baseline'])} {u}<extra></extra>")
    fig.add_scatter(x=[ex["prediction"]], y=["Prediction"], mode="markers+text", showlegend=False, cliponaxis=False,
                    text=[f"  {num(ex['prediction'])} {u}"], textposition="middle right",
                    marker=dict(size=12, color=c["blue"], line=dict(width=2, color=c["surface"])),
                    hovertemplate=f"Prediction = starting point + all pushes<br>{num(ex['prediction'])} {u}<extra></extra>")
    pts = [ex["baseline"], ex["prediction"], *ends]
    if observed is not None:
        pts.append(observed)
        fig.add_vline(x=observed, line_color=c["aqua"], line_width=1.5,
                      annotation_text=f"Measured: {num(observed)} {u}", annotation_position="top",
                      annotation_font_size=11, annotation_yshift=6)
    lo, hi = min(pts), max(pts)
    span = hi - lo or 1.0
    style(fig, f"{outcome(b)} ({u})", "", height=110 + 44 * len(rows), legend=False)
    fig.update_xaxes(range=[lo - 0.15 * span, hi + 0.3 * span])
    fig.update_yaxes(categoryorder="array", categoryarray=rows, autorange="reversed", showgrid=False)
    fig.update_layout(margin=dict(t=44))
    show(fig, key="linear_waterfall")


def _category_breakdown(b: dict, values: dict, ex: dict) -> None:
    lin, u = b["linear"], b["units"]
    with st.expander("Method note: how a category (sex, ethnicity) becomes one push", icon=":material/category:"):
        st.markdown("The model stores each category as 0/1 columns, one per group except a **reference group** "
                    "that is all zeros. Each column gets its own term, coefficient × (0 or 1 − training share), "
                    "and the terms are added into one push. Because the training share is an average mix, even a "
                    "person in the reference group gets a small push. That is correct, not an error.")
        for g in b["categorical_groups"]:
            cols = [lv for lv in g["levels"] if lv["column"]]
            rows = [{"Column (group)": lv["label"], f"Coefficient vs {g['reference']} ({u})": val(lin["coefficients"][lv["column"]]),
                     "Training share": f"{lin['reference'][lv['column']]:.3f}", "This person": f"{values[lv['column']]:g}",
                     f"Term ({u})": signed(ex["pushes"][lv["column"]], 2)} for lv in cols]
            st.markdown(f"**{g['label']}** (reference group: {g['reference']}): "
                        f"push = **{signed(ex['grouped_pushes'][g['name']], 2)} {u}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---- 06. Quantitative interpretation -------------------------------------------
def _coefficient_vs_push(b: dict, person: dict, values: dict, ex: dict) -> None:
    c, u, lin = palette(), b["units"], b["linear"]
    subsection("6.1", "A coefficient is a slope; a push is one person's position on it")
    numeric = [f["name"] for f in numeric_features(b)]
    default = max(numeric, key=lambda n: abs(ex["pushes"][n]))
    with figure("2", "Coefficient versus contribution",
                "Push as a function of one feature's value; the slope is the coefficient"):
        name = st.selectbox("Feature", numeric, index=numeric.index(default),
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
                        mode="markers+text", name="Other validation subjects (recorded values)",
                        text=[p["label"].replace("Example person ", "P") for p in others], textposition="top center",
                        textfont=dict(size=10), marker=dict(size=8, color=c["muted"], line=dict(width=2, color=c["surface"])),
                        hovertemplate=f"value %{{x:.3g}}<br>push %{{y:+.2f}} {u}<extra></extra>")
        fig.add_scatter(x=[x], y=[push], mode="markers", name=f"{person['label']}: push {signed(push)} {u}",
                        marker=dict(size=13, color=c["orange"], line=dict(width=2, color=c["surface"])),
                        hovertemplate=f"value %{{x:.3g}}<br>push %{{y:+.2f}} {u}<extra></extra>")
        show(style(fig, f"{f['label']} ({f['units']})", f"Push ({u})", height=400), key="linear_slope")
        fig_caption("2", "The line is the push the model assigns to every possible value of this feature. It passes "
                         "through zero at the training average, and its slope is the coefficient. Points are the five "
                         "validation subjects: they share one slope but sit at different positions, so they receive "
                         "different pushes. The range shown covers the 1st–99th percentile of training values.")

    st.latex(rf"\text{{push}} = \beta \times (x - \bar x) = {val(coef)} \times ({val(x)} - {val(ref)}) "
             rf"= {signed(push, 2)}\ \text{{{u}}}")
    side = "above" if x > ref else "below" if x < ref else "exactly at"
    st.markdown(f"The **coefficient** ({val(coef)} {u} per {per}) is the slope of the line. It is the same for all "
                f"five validation subjects, and for everyone else. The **push** depends on where one person sits: "
                f"{person['label']}'s value is {side} the training average, so the push is **{signed(push)} {u}**.")
    for g in b["categorical_groups"]:
        _category_text(b, g, values, ex)


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
                    f"**reference group**, *{g['reference']}*. That is only how the 0/1 column is coded.\n"
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
    subsection("6.2", "General effect size versus subject-specific contribution")
    st.markdown("Raw coefficients cannot be compared across features, because each is *per one unit* of a different "
                "quantity (per %, per year, per cm). Multiplying a coefficient by the feature's training standard "
                "deviation (SD) gives the change in prediction for a *typical* difference between people: a fair, "
                "general comparison. A push is one person's contribution: a feature can matter a lot in general and "
                "still give a small push to someone near the average.")
    names = sorted(lin["typical_effect_per_sd"], key=lambda n: -abs(lin["typical_effect_per_sd"][n]))
    labels = [short_label(b, n) for n in names]
    with figure("3", "Standardized effect size and subject contribution",
                f"Change in predicted {outcome(b, True)} ({u}): per 1-SD difference (all people) and this subject's push"):
        fig = go.Figure()
        fig.add_bar(y=labels, x=[lin["typical_effect_per_sd"][n] for n in names], orientation="h",
                    name="Effect of a 1-SD difference (coefficient × SD)", marker=dict(color=c["blue"], cornerradius=2),
                    hovertemplate=f"%{{y}}<br>1-SD effect %{{x:+.2f}} {u}<extra></extra>")
        fig.add_bar(y=labels, x=[ex["pushes"][n] for n in names], orientation="h",
                    name=f"{person['label']}: push", marker=dict(color=c["orange"], cornerradius=2),
                    hovertemplate=f"%{{y}}<br>this person's push %{{x:+.2f}} {u}<extra></extra>")
        fig.add_vline(x=0, line_color=c["muted"], line_width=1)
        style(fig, f"Change in predicted {outcome(b, True)} ({u})", "", height=120 + 58 * len(names))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig, key="linear_general")
        fig_caption("3", "Blue: a property of the model, the same for everyone. Orange: a property of this subject. "
                         "Features are ordered by standardized effect size. Sex and ethnicity are omitted because a "
                         "0/1 category has no typical 1-SD difference; their coefficients are group comparisons (§6.1).")
        table_view(pd.DataFrame([{
            "Feature": label_of(b, n), f"Coefficient ({u} per unit)": val(lin["coefficients"][n]),
            "Training SD": val(b["feature_stats"][n]["train_sd"]),
            f"Typical effect per SD ({u})": signed(lin["typical_effect_per_sd"][n], 2),
            f"This person's push ({u})": signed(ex["pushes"][n], 2)} for n in names]))


def _intercept(b: dict) -> None:
    lin, u = b["linear"], b["units"]
    subsection("6.3", "Intercept versus reference prediction")
    terms = {f: lin["coefficients"][f] * lin["reference"][f] for f in lin["coefficients"]}
    zero = [f"{short_label(b, f['name'])} 0{unit(f['units'])}" for f in numeric_features(b)]
    zero += [g["reference"].lower() if g["name"] == "sex" else g["reference"] for g in b["categorical_groups"]]
    st.markdown(f"The **intercept** $b_0$ = **{num(lin['intercept'], 2)} {u}** is the prediction when *every* "
                f"feature equals 0: {', '.join(zero)}. No such person exists, so the intercept is "
                "an extrapolation far outside the data. The **starting point** moves from that impossible person to "
                "the training average:")
    st.latex(rf"\text{{starting point}} = b_0 + \sum_j \beta_j \bar x_j = {num(lin['intercept'], 2)} "
             rf"{'+' if sum(terms.values()) >= 0 else '-'} {num(abs(sum(terms.values())), 2)} "
             rf"= {num(lin['baseline'], 2)}\ \text{{{u}}}")
    table_view(pd.DataFrame([{"Feature": label_of(b, f), "Coefficient β": val(lin["coefficients"][f]),
                              "Training average x̄": val(lin["reference"][f]), f"β × x̄ ({u})": num(t, 2)}
                             for f, t in terms.items()]), "Data table: each β × x̄ term")
    st.caption("Same model, same predictions: only the reference point differs. Pushes measured from the training "
               "average stay small and readable; 'pushes' measured from zero would be large and meaningless.")


# ---- 07. Scientific interpretation ---------------------------------------------
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
    subsection("7.1", "Residual diagnostics")
    rmse = lin["rmse"]
    st.markdown(f"A **residual** is measured − predicted. On the {b['n_validation']} validation participants (never "
                f"used for fitting), the typical residual size, the **root-mean-square error (RMSE)**, is "
                f"**{num(rmse['validation'])} {u}** (training rows: {num(rmse['train'])} {u}).")
    if observed is not None:
        res = observed - ex["prediction"]
        st.markdown(f"For {person['label']}: {num(observed)} − {num(ex['prediction'])} = **{signed(res)} {u}**, "
                    f"about {abs(res) / rmse['validation']:.1f}× the typical error.")
        if abs(res) > 1.5 * rmse["validation"]:
            note("Model limitation", f"The line misses {person['label']} by a lot: it predicts "
                 f"{ex['prediction'] / observed:.1f}× the measured value. That is the model, not a bug. A single "
                 "straight line has one slope per feature for everyone, so it cannot follow a relationship that "
                 "bends. Fig. 4 shows where that happens.")

    rows = _validation_rows(b["id"])
    numeric = [f["name"] for f in numeric_features(b)]
    with figure("4", "Residual structure across the validation cohort",
                f"{b['n_validation']} validation participants · fixed linear model"):
        name = st.selectbox("Feature", numeric, index=numeric.index(FOCUS_FEATURE.get(b["id"], numeric[0])),
                            format_func=lambda n: label_of(b, n), key=f"resid_{b['id']}")
        f = feature(b, name)
        me = rows[rows.row_id == person["row_id"]] if observed is not None else rows.iloc[0:0]
        tab1, tab2 = st.tabs(["(a) Measured vs. feature, with the model's line", "(b) Residuals vs. feature"])
        with tab1:
            grid = np.linspace(rows[name].min(), rows[name].max(), 60)
            line = [m10.linear_explain(b, {**lin["reference"], name: g})["prediction"] for g in grid]
            fig = go.Figure()
            fig.add_scatter(x=rows[name], y=rows[b["target"]], mode="markers", name="Validation participants (measured)",
                            marker=dict(size=5, color=c["muted"], opacity=0.45),
                            hovertemplate=f"{f['label']} %{{x}}<br>measured %{{y:.1f}} {u}<extra></extra>")
            fig.add_scatter(x=grid, y=line, mode="lines", name="Model prediction (other features at training average)",
                            line=dict(color=c["blue"], width=2.5))
            if len(me):
                fig.add_scatter(x=me[name], y=me[b["target"]], mode="markers", name=person["label"],
                                marker=dict(size=13, color=c["orange"], line=dict(width=2, color=c["surface"])))
            show(style(fig, f"{f['label']} ({f['units']})", f"Measured {outcome(b, True)} ({u})", height=420), key="linear_fit")
        with tab2:
            bins = pd.qcut(rows[name], 10, duplicates="drop")
            binned = rows.groupby(bins, observed=True)[[name, "residual"]].mean()
            fig = go.Figure()
            fig.add_hline(y=0, line_color=c["muted"], line_width=1)
            fig.add_scatter(x=rows[name], y=rows["residual"], mode="markers", name="Validation participants",
                            marker=dict(size=5, color=c["muted"], opacity=0.45),
                            hovertemplate=f"{f['label']} %{{x}}<br>residual %{{y:+.1f}} {u}<extra></extra>")
            fig.add_scatter(x=binned[name], y=binned["residual"], mode="lines+markers",
                            name="Mean residual (deciles of the feature)", line=dict(color=c["violet"], width=2.5),
                            marker=dict(size=7, line=dict(width=2, color=c["surface"])))
            if len(me):
                fig.add_scatter(x=me[name], y=me["residual"], mode="markers", name=person["label"],
                                marker=dict(size=13, color=c["orange"], line=dict(width=2, color=c["surface"])))
            show(style(fig, f"{f['label']} ({f['units']})", f"Residual: measured − predicted ({u})", height=420),
                 key="linear_resid")
        fig_caption("4", "(a) Measured outcomes of the validation participants against one feature, with the model's "
                         "prediction along that feature when every other feature is held at its training average. "
                         "(b) Residuals against the same feature; the violet line is the mean residual within each "
                         "tenth of the feature's range. If a straight line were adequate, residuals would scatter "
                         "evenly around zero and the violet line would stay flat. A trend or curve is systematic "
                         "error: structure the model cannot represent, not noise.")
    if b["id"] == "fev1":
        note("Interpretation", "Lung function shows this clearly against age. Lungs grow through childhood and shrink "
             "later in life, but the model has one slope for age. So it over-predicts for children (residuals below "
             "zero), under-predicts for young adults and over-predicts again for older adults. Height, which would "
             "explain much of this, was deliberately left out of the data.")


PERSPECTIVE = """
**Problem statement.** Can structure-derived descriptors of capsid subunit interfaces predict a continuous
biophysical quantity, such as the **interface binding free energy** (ΔG, kcal/mol)? An AAV capsid is a T = 1
icosahedral shell of 60 subunits, with distinct interfaces around its 2-fold, 3-fold and 5-fold symmetry axes.
Each interface, in each characterized variant, could be described by buried surface area, hydrogen bonds, salt
bridges, shape complementarity and the local resolution of the cryo-EM map around it.

**Formal correspondence.** A linear model of such a table would be read exactly as in this module:

| This module (NHANES) | Structural analogue (hypothetical) |
|---|---|
| Validation subject | One interface held out from fitting |
| Feature $x_j$ (e.g. waist circumference, cm) | Descriptor (e.g. buried surface area, Å²) |
| Training average $\\bar x_j$ | Mean descriptor over the training interfaces |
| Coefficient $\\beta_j$ | Sensitivity: change in predicted ΔG per unit of descriptor, others held fixed |
| Push $\\beta_j(x_j-\\bar x_j)$ | Signed contribution of the descriptor relative to the reference interface |
| Starting point | Predicted ΔG of a reference (all-average) interface |
| Intercept $b_0$ | Extrapolation to an interface with every descriptor at zero: non-physical |
| Reference group (0/1 coding) | Reference interface class, e.g. 2-fold against 3-fold and 5-fold |
| Residual | Measured − predicted ΔG |

**Analytical cautions.**
1. *Collinearity.* Buried area, hydrogen-bond count and contact number rise together. Their individual
   coefficients become unstable and "holding the others fixed" can describe interfaces that never occur. Report
   effects per SD and check variance inflation before reading any single coefficient.
2. *Measurement error.* Descriptors read from cryo-EM maps carry error that varies with local resolution. Random
   error in a predictor tends to shrink its coefficient toward zero (regression dilution).
3. *Non-independence.* The 60 copies of a subunit make symmetry-related interfaces near-replicates. The unit of
   analysis should be the unique interface type per variant, not each copy, or the effective sample size is
   overstated.
4. *Association, not mechanism.* A positive coefficient on buried area does not show that adding buried area
   would strengthen binding. Causal claims need designed perturbations, such as alanine-scanning or charge-swap
   mutants with measured stability (for example by differential scanning fluorimetry) or assembly readouts.

**Diagnostic to report.** Residuals against each descriptor, as in Fig. 4. A curved residual pattern, such as
large interfaces being systematically under-predicted, is evidence of non-linearity, the same lesson FEV₁ against
age teaches here. It calls for a better model, not a hidden plot.
"""
