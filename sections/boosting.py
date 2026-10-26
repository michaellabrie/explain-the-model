"""Module 3 (One Correction at a Time): gradient boosting adds one small, learning-rate-scaled tree per round.

Round 0 = boosting.init (the training average); staged[m] = staged[m-1] + corrections[m-1].
The corrections already include the learning rate, so they are never multiplied by it again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sections.common import (FOCUS_FEATURE, feature, feature_profile, fig_caption, figure, hline, label_of, load_data,
                             note, num, numeric_features, outcome, page_header, palette, pick_person,
                             research_translation, section, show, signed, style, subject_record, subsection, table,
                             vline)

VIEWS = {"First 100 rounds": 100, "All 500 rounds": 500}


def page() -> None:
    b, person = pick_person()
    u, gb = b["units"], b["boosting"]
    staged = gb["people"][person["id"]]["staged"]
    m, lr, depth = len(staged) - 1, gb["learning_rate"], gb["settings"]["max_depth"]
    page_header(b, 3, "One Correction at a Time", "Gradient boosting",
                "Stagewise Residual Correction by Gradient Boosting",
                "How does each new tree improve the model, and when should training stop?",
                f"Gradient boosting regressor · {m} rounds · learning rate {lr} · depth-{depth} trees · "
                "squared-error loss")

    section(1)
    st.markdown(
        f"**Question.** How does gradient boosting build its prediction of {outcome(b, True)} ({u}) one small correction "
        f"at a time, and how should the number of corrections be chosen?\n\n"
        f"**Learning objective.** Explain how boosting adds one small correction per round, and use validation error "
        f"to choose when to stop.\n\n"
        f"**Approach.** Trace one subject's prediction from round 0 to round {m}, inspect the training residuals each "
        f"new tree is fitted to, and compare training with validation error to locate the point where further "
        f"rounds stop helping new people.")

    section(2, "The boosting recursion")
    _algorithm(b)

    section(3)
    subject_record(b, person)

    section(4, "Feature vector")
    table("1", "Standardized feature profile", feature_profile(b, person["values"]),
          "z = (subject value − training mean) / training SD. Values are the subject's recorded survey values. "
          "Editing feature values is part of Module 1; this module shows the supplied model outputs only.")

    section(5, "Round-by-round prediction trajectory")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Round 0 ({u})", num(staged[0]), help="Round 0 is the average outcome in the training rows.")
    c2.metric(f"After round 1 ({u})", num(staged[1]))
    c3.metric(f"After round {m} ({u})", num(staged[m]))
    c4.metric(f"Measured ({u})", num(person["observed"]))
    _one_person(b, person)

    section(6, "Residual targets, error curves and step size")
    _residuals(b)
    _stopping(b)
    _learning_rate(b, person)

    section(7, "Overfitting, validation-based stopping and model choice")
    _interpretation(b)
    _compare(b)
    research_translation("iterative refinement and the held-out criterion",
                         "How residual-driven, held-out-validated refinement in boosting parallels model refinement in "
                         "crystallography and cryo-EM, and how a structural dataset would need to be split to make the "
                         "validation curve honest.", PERSPECTIVE)


# ---- 02. Recursion ---------------------------------------------------------------
def _algorithm(b: dict) -> None:
    u, gb = b["units"], b["boosting"]
    depth, rounds, lr = gb["settings"]["max_depth"], gb["settings"]["n_estimators"], gb["learning_rate"]
    st.latex(rf"""\begin{{aligned}}
F_0(x) &= \bar y_{{\text{{train}}}} = {num(gb['init'], 2)}\ \text{{{u}}} \\
\text{{for }} m &= 1, \dots, {rounds}: \\
r_i &= y_i - F_{{m-1}}(x_i) && \text{{residuals on the training rows}} \\
h_m &= \text{{a depth-{depth} tree fitted to }} (x_i,\ r_i) \\
F_m(x) &= F_{{m-1}}(x) + \underbrace{{\nu\, h_m(x)}}_{{\text{{correction}}}}, \qquad \nu = {lr}
\end{{aligned}}""")
    st.markdown(
        "| Symbol | Meaning | In this dataset |\n|---|---|---|\n"
        f"| $F_0$ | **Round 0**: the training average, the best single constant guess under squared error | "
        f"{num(gb['init'])} {u} |\n"
        "| $r_i$ | **Residual** of training row $i$: measured − current prediction, what is still unexplained | "
        "Fig. 3 |\n"
        f"| $h_m$ | **Tree $m$**, fitted to the residuals rather than to the outcome. Depth {depth}: at most {depth} "
        f"questions and {2 ** depth} leaves, a deliberately weak model | {rounds} trees |\n"
        f"| $\\nu$ | **Learning rate**: each tree's output is shrunk to {lr:.0%} before it is added | {lr} |\n"
        "| $\\nu h_m(x)$ | **Correction** added in round $m$. The stored corrections already include $\\nu$ and are "
        "never multiplied by it again | Fig. 2 |\n"
        f"| $F_m(x)$ | Prediction after round $m$ | Fig. 1 |")
    st.markdown("**Why \"gradient\"?** For squared-error loss $L = \\tfrac12 (y - F)^2$, the slope of the loss with "
                "respect to the prediction is $-(y - F)$. The residual is therefore the *negative gradient*, so each "
                "tree is a small step downhill on the training error, taken in the space of prediction functions "
                "rather than of parameters.")


# ---- 05. Trajectory --------------------------------------------------------------
def _one_person(b: dict, person: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    staged, corr = gb["people"][person["id"]]["staged"], gb["people"][person["id"]]["corrections"]
    m, lr, best = len(corr), gb["learning_rate"], gb["best_validation_round"]
    with figure("1", f"Prediction trajectory for {person['label']}",
                f"Predicted {outcome(b, True)} after each round ({u}), with the measured value and the validation-selected round"):
        r = st.slider("Round", 0, m, 1, key="gb_round")
        if r == 0:
            st.latex(rf"F_0 = \bar y_{{\text{{train}}}} = {num(staged[0], 2)}\ \text{{{u}}}")
        else:
            cr = corr[r - 1]
            st.latex(rf"F_{{{r}}} = F_{{{r - 1}}} + \nu\,h_{{{r}}}(x) = {num(staged[r - 1], 2)} "
                     rf"{'+' if cr >= 0 else '-'} {abs(cr):.2f} = {num(staged[r], 2)}\ \text{{{u}}}")
        view = st.radio("Rounds shown", list(VIEWS), horizontal=True, key="gb_view_person")
        hi = min(VIEWS[view], m)
        xs = np.arange(hi + 1)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=staged[:hi + 1], mode="lines", name="Prediction after each round",
                        line=dict(color=c["blue"], width=2), hovertemplate=f"round %{{x}}: %{{y:.1f}} {u}<extra></extra>")
        hline(fig, person["observed"], f"Measured: {num(person['observed'])}", c["aqua"], 0, hi, 1.5)
        lo_y, hi_y = min(min(staged[:hi + 1]), person["observed"]), max(max(staged[:hi + 1]), person["observed"])
        if best <= hi:
            vline(fig, best, f"Lowest validation RMSE (round {best})", c["muted"], lo_y, hi_y, 1.5)
        if r <= hi:
            fig.add_scatter(x=[r], y=[staged[r]], mode="markers", name=f"Round {r}: {num(staged[r])}",
                            marker=dict(size=11, color=c["ink"], line=dict(width=2, color=c["surface"])))
        show(style(fig, "Round (number of trees added)", f"Predicted {outcome(b, True)} ({u})", height=360), key="gb_path")
        fig_caption("1", f"The prediction starts at the training average (round 0) and moves by one correction per "
                         f"round. The marker follows the slider. {person['label']} is a validation row, so no tree "
                         "ever saw their residual: the corrections they receive come from patterns the trees learned "
                         "on training rows with similar feature values.")
    if r > 0:
        st.markdown(f"Tree {r}'s own output for this subject was $h_{{{r}}}(x)$ = {signed(corr[r - 1] / lr, 2)} {u}; "
                    f"the model adds only $\\nu$ × that, a correction of **{signed(corr[r - 1], 2)} {u}**. (That stored "
                    f"correction already includes $\\nu$ = {lr}.)")
    else:
        st.markdown("Round 0: no trees yet, so the prediction is the average outcome in the training rows.")

    early, late = np.mean(np.abs(corr[:10])), np.mean(np.abs(corr[100:]))
    with figure("2", "Correction added by each tree",
                f"Change in {person['label']}'s prediction in each round ({u}), already scaled by the learning rate"):
        fig = go.Figure()
        fig.add_bar(x=xs[1:], y=corr[:hi], marker=dict(color=[c["red"] if v > 0 else c["blue"] for v in corr[:hi]]),
                    hovertemplate=f"tree %{{x}} added %{{y:+.3f}} {u}<extra></extra>", showlegend=False)
        fig.add_hline(y=0, line_color=c["muted"], line_width=1)
        style(fig, "Round (tree number)", f"Correction ({u})", height=260, legend=False)
        fig.update_layout(bargap=0.15)
        show(fig, key="gb_corrections")
        fig_caption("2", f"Red: upward corrections; blue: downward. The mean absolute correction is {early:.2f} {u} in "
                         f"rounds 1–10 and {late:.3f} {u} in rounds 101–{m}; together, rounds 101–{m} move this "
                         f"subject's prediction by {signed(staged[m] - staged[100])} {u}. Corrections shrink because "
                         "the training residuals the trees chase shrink.")


# ---- 06. Quantitative interpretation -------------------------------------------
def _residuals(b: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    snap = gb["residual_snapshots"]
    subsection("6.1", "Residual targets for successive trees")
    st.markdown("Each tree is fitted to the residuals (measured − current prediction) on the **training** rows. Any "
                "pattern left in the residuals, such as residuals that are mostly positive over one range of a "
                "feature, is what the next tree will try to capture. When no pattern is left, there is little for "
                "new trees to learn except noise.")
    rounds = snap["rounds"]
    with figure("3", "Training residuals at selected rounds",
                f"{len(snap['row_ids'])} training rows · residual = measured − prediction after the chosen round ({u})"):
        r = st.select_slider("Round", rounds, value=rounds[0], key="gb_snap")
        obs = np.array(snap["observed"])
        pred = np.array(snap["predictions"][str(r)])
        res = obs - pred
        lim = 1.08 * np.max(np.abs(obs - np.array(snap["predictions"][str(rounds[0])])))
        rows = load_data(b["id"]).set_index("row_id").loc[snap["row_ids"]]
        numeric = [f["name"] for f in numeric_features(b)]
        name = st.selectbox("Feature", numeric, index=numeric.index(FOCUS_FEATURE.get(b["id"], numeric[0])),
                            format_func=lambda n: label_of(b, n), key=f"gb_feat_{b['id']}")
        f = feature(b, name)
        x = rows[name].to_numpy()
        tab1, tab2 = st.tabs(["(a) Residuals vs. a feature", "(b) Predicted vs. measured"])
        with tab1:
            frame = pd.DataFrame({"x": x, "res": res})
            binned = frame.groupby(pd.qcut(frame.x, 8, duplicates="drop"), observed=True).mean()
            fig = go.Figure()
            fig.add_hline(y=0, line_color=c["muted"], line_width=1)
            fig.add_scatter(x=x, y=res, mode="markers", name="Training rows", marker=dict(size=6, color=c["blue"], opacity=0.5),
                            hovertemplate=f"{f['label']} %{{x}}<br>residual %{{y:+.1f}} {u}<extra></extra>")
            fig.add_scatter(x=binned.x, y=binned.res, mode="lines+markers", name="Mean residual (eighths of the feature)",
                            line=dict(color=c["violet"], width=2.5), marker=dict(size=7, line=dict(width=2, color=c["surface"])))
            style(fig, f"{f['label']} ({f['units']})", f"Residual after round {r} ({u})", height=400)
            fig.update_yaxes(range=[-lim, lim])
            show(fig, key="gb_resid")
        with tab2:
            lo, hi = float(obs.min()), float(obs.max())
            fig = go.Figure()
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Identity (measured = predicted)",
                            line=dict(color=c["muted"], width=1.5))
            fig.add_scatter(x=pred, y=obs, mode="markers", name="Training rows", marker=dict(size=6, color=c["blue"], opacity=0.5),
                            hovertemplate=f"predicted %{{x:.1f}}<br>measured %{{y:.1f}} {u}<extra></extra>")
            style(fig, f"Predicted after round {r} ({u})", f"Measured ({u})", height=400)
            fig.update_xaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
            show(fig, key="gb_predobs")
        tr = gb["rmse"]["train"]
        fig_caption("3", f"(a) Residuals against one feature, on a fixed vertical scale so that shrinkage is visible; "
                         f"the violet line is the mean residual within each eighth of the feature's range. (b) The "
                         f"same rows as predicted against measured; points approach the identity line as rounds "
                         f"accumulate. Training RMSE over all {b['n_train']} training rows: {num(tr[r])} {u} after "
                         f"round {r}, against {num(tr[0])} {u} at round 0, which is simply the spread of the outcome "
                         "itself, because everyone gets the same prediction.")


def _stopping(b: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    tr, va = gb["rmse"]["train"], gb["rmse"]["validation"]
    best, best_rmse, m = gb["best_validation_round"], gb["best_validation_rmse"], len(tr) - 1
    subsection("6.2", "Training versus validation error")
    st.markdown(f"**Training error** is measured on the rows the trees are fitted to, so it keeps falling: every tree "
                f"is built to reduce it. **Validation error** is measured on the {b['n_validation']} held-out rows "
                f"that no tree was fitted to, and shows whether the corrections still help new people.")
    with figure("4", "Learning curves", f"Root-mean-square error ({u}) after each round, on training and validation rows"):
        stop = st.slider("Stop after round", 0, m, m, key="gb_stop")
        view = st.radio("Rounds shown", list(VIEWS), index=1, horizontal=True, key="gb_view_rmse")
        hi = min(VIEWS[view], m)
        xs = np.arange(hi + 1)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=tr[:hi + 1], mode="lines", name="Training RMSE", line=dict(color=c["blue"], width=2),
                        hovertemplate=f"round %{{x}}: training %{{y:.2f}} {u}<extra></extra>")
        fig.add_scatter(x=xs, y=va[:hi + 1], mode="lines", name="Validation RMSE", line=dict(color=c["orange"], width=2),
                        hovertemplate=f"round %{{x}}: validation %{{y:.2f}} {u}<extra></extra>")
        lo_y, hi_y = min(tr[:hi + 1]), max(max(tr[:hi + 1]), max(va[:hi + 1]))
        vline(fig, best, f"Lowest validation RMSE: round {best}", c["muted"], lo_y, hi_y, 1.5)
        if stop <= hi:
            fig.add_scatter(x=[stop, stop], y=[tr[stop], va[stop]], mode="markers", name=f"Chosen stop: round {stop}",
                            marker=dict(size=11, color=c["ink"], line=dict(width=2, color=c["surface"])))
        show(style(fig, "Round (number of trees)", f"RMSE ({u})", height=380), key="gb_rmse")
        fig_caption("4", "Blue: error on the rows the trees were fitted to. Orange: error on held-out validation rows. "
                         "The grey line marks the round with the lowest validation error; the dark markers show the "
                         "round chosen with the slider. Units are those of the outcome; RMSE values are not comparable "
                         "across datasets.")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Training RMSE at round {stop}", f"{num(tr[stop])} {u}")
    m2.metric(f"Validation RMSE at round {stop}", f"{num(va[stop])} {u}", delta=f"{signed(va[stop] - best_rmse)} {u} vs best",
              delta_color="inverse")
    m3.metric("Lowest validation RMSE", f"{num(best_rmse)} {u}", delta=f"at round {best}", delta_color="off",
              delta_arrow="off")


def _learning_rate(b: dict, person: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    cmp = gb["comparison_learning_rate_0_02"]
    lr, best, cbest = gb["learning_rate"], gb["best_validation_round"], cmp["best_validation_round"]
    va, cva = gb["rmse"]["validation"], cmp["rmse"]["validation"]
    staged, cstaged = gb["people"][person["id"]]["staged"], cmp["people"][person["id"]]["staged"]
    m = len(va) - 1
    subsection("6.3", "Effect of the learning rate")
    st.markdown(f"The learning rate sets how much of each tree's output is added. Here the main model ($\\nu$ = {lr}) is "
                f"compared with a **separately fitted** model with $\\nu$ = 0.02. The 0.02 curves come from that model, "
                f"not from shrinking the 0.1 curves.")
    xs = np.arange(m + 1)
    with figure("5", "Learning-rate comparison",
                f"(a) Validation RMSE ({u}) · (b) {person['label']}'s predicted {outcome(b, True)} ({u})"):
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_scatter(x=xs, y=va, mode="lines", name=f"learning rate {lr}", line=dict(color=c["orange"], width=2),
                            hovertemplate=f"round %{{x}}: %{{y:.2f}} {u}<extra></extra>")
            fig.add_scatter(x=xs, y=cva, mode="lines", name="learning rate 0.02 (separate model)", line=dict(color=c["violet"], width=2),
                            hovertemplate=f"round %{{x}}: %{{y:.2f}} {u}<extra></extra>")
            fig.add_scatter(x=[best, cbest], y=[va[best], cva[cbest]], mode="markers", name="Lowest point of each",
                            marker=dict(size=9, color=c["muted"], line=dict(width=2, color=c["surface"])))
            show(style(fig, "Round", f"(a) Validation RMSE ({u})", height=340), key="gb_lr_rmse")
        with right:
            fig = go.Figure()
            fig.add_scatter(x=xs, y=staged, mode="lines", name=f"learning rate {lr}", line=dict(color=c["blue"], width=2),
                            hovertemplate=f"round %{{x}}: %{{y:.1f}} {u}<extra></extra>")
            fig.add_scatter(x=xs, y=cstaged, mode="lines", name="learning rate 0.02 (separate model)", line=dict(color=c["violet"], width=2),
                            hovertemplate=f"round %{{x}}: %{{y:.1f}} {u}<extra></extra>")
            hline(fig, person["observed"], "Measured", c["aqua"], 0, m, 1.5)
            show(style(fig, "Round", f"(b) Predicted {outcome(b, True)} ({u})", height=340), key="gb_lr_person")
        same = abs(cmp["best_validation_rmse"] - gb["best_validation_rmse"]) < 0.01 * gb["best_validation_rmse"]
        fig_caption("5", f"With learning rate 0.02 the validation error bottoms out at round {cbest} instead of {best}, "
                         f"about {cbest / best:.0f}× as many rounds, "
                         f"{'reaching essentially the same lowest error' if same else 'with a lowest error of'} "
                         f"({num(cmp['best_validation_rmse'])} {u} vs {num(gb['best_validation_rmse'])} {u} for "
                         f"{lr}). Smaller steps take more rounds to get to the same place; in exchange the error curve "
                         "is typically flatter near its lowest point, so the result is less sensitive to exactly where "
                         "training stops.")
    table("2", "Lowest validation error by learning rate", pd.DataFrame([
        {"Learning rate": str(lr), f"Lowest validation RMSE ({u})": num(gb["best_validation_rmse"]),
         "At round": best, f"This subject then ({u})": num(staged[best]), f"Round {m} ({u})": num(staged[m])},
        {"Learning rate": "0.02 (separate)", f"Lowest validation RMSE ({u})": num(cmp["best_validation_rmse"]),
         "At round": cbest, f"This subject then ({u})": num(cstaged[cbest]), f"Round {m} ({u})": num(cstaged[m])}]),
          "Both stopping rounds are validation-based choices.")


# ---- 07. Scientific interpretation ---------------------------------------------
def _interpretation(b: dict) -> None:
    u, gb = b["units"], b["boosting"]
    tr, va = gb["rmse"]["train"], gb["rmse"]["validation"]
    best, best_rmse, m = gb["best_validation_round"], gb["best_validation_rmse"], len(tr) - 1
    subsection("7.1", "Overfitting and validation-based stopping")
    st.markdown(f"The lowest validation RMSE is **{num(best_rmse)} {u} at round {best}**. After that, training RMSE "
                f"keeps falling ({num(tr[best])} → {num(tr[m])} {u} by round {m}) while validation RMSE rises "
                f"({num(va[best])} → {num(va[m])} {u}). The later trees are fitting quirks of the training rows that do "
                f"not recur in new people: **overfitting**.")
    note("Interpretation", f"Stopping at round {best} is a **validation-based choice**, not a test result. The validation "
         f"rows were used to pick the round, so {num(best_rmse)} {u} is slightly optimistic. An honest estimate of error "
         f"on new people would need a separate test set that never influenced any decision.")


def _compare(b: dict) -> None:
    u, lin, rf, gb = b["units"], b["linear"], b["forest"], b["boosting"]
    tr, best = gb["rmse"]["train"], gb["best_validation_round"]
    m = len(tr) - 1
    subsection("7.2", "Model comparison within this dataset")
    fair = {"Linear regression": lin["rmse"]["validation"], f"Random forest ({rf['n_trees']} trees)": rf["rmse"]["validation"],
            f"Gradient boosting ({m} rounds)": gb["rmse"]["validation"][m]}
    table("3", "Training and validation error of the three fixed models", pd.DataFrame([
        {"Model": "Linear regression", f"Training RMSE ({u})": num(lin["rmse"]["train"]),
         f"Validation RMSE ({u})": num(lin["rmse"]["validation"]), "Note": "one straight-line equation"},
        {"Model": f"Random forest ({rf['n_trees']} trees)", f"Training RMSE ({u})": num(rf["rmse"]["train"]),
         f"Validation RMSE ({u})": num(rf["rmse"]["validation"]), "Note": "average of fully grown trees"},
        {"Model": f"Gradient boosting ({m} rounds)", f"Training RMSE ({u})": num(tr[m]),
         f"Validation RMSE ({u})": num(gb["rmse"]["validation"][m]), "Note": "the full fitted model"},
        {"Model": f"Gradient boosting stopped at round {best}", f"Training RMSE ({u})": num(tr[best]),
         f"Validation RMSE ({u})": num(gb["best_validation_rmse"]),
         "Note": "round chosen with these same validation rows, so optimistic"}]),
          "Compare within this dataset only: each dataset has its own units.")
    winner = min(fair, key=fair.get)
    st.markdown(f"Among the three fixed models as fitted, the lowest validation RMSE here belongs to **{winner}**, "
                f"at {num(fair[winner])} {u}. When the flexible tree models do not beat a straight line, the "
                "relationship is close to linear, or the features carry little extra signal. When they beat it "
                "clearly, there is non-linear structure the line misses. The gap between the training and validation "
                "columns shows how much better each model fits its own training rows than new people.")


PERSPECTIVE = """
**Problem statement.** Consider predicting a continuous property of AAV capsid variants, such as assembly yield or
an interface binding free energy (ΔG, kcal/mol), from structure-derived descriptors at the 3-fold and 5-fold
interfaces. Boosting would start every variant at the training mean. Early trees would correct the largest, broadest
errors (perhaps a whole class of interfaces); later trees would make smaller, more specific adjustments.

**Formal correspondence.** Boosting is iterative refinement guided by residuals and policed by held-out data,
which is the logic of macromolecular model refinement:

| This module (NHANES) | Structural analogue |
|---|---|
| Round-0 prediction $F_0$ | Starting model |
| Residuals $r_i$ on training rows | Difference density (e.g. an $F_o - F_c$ map): what the current model does not yet explain |
| Tree $h_m$ fitted to the residuals | A local correction built into that unexplained signal |
| Learning rate $\\nu$ | Step size or damping of each refinement cycle |
| Training RMSE | R-work: agreement with the data the model is refined against |
| Validation RMSE | R-free: agreement with reflections held out from refinement |
| Lowest validation round | Stopping when R-free stops improving, even if R-work still falls |
| Independent test set | Validation against data that never influenced any modelling choice |

In cryo-EM the same safeguard appears as gold-standard refinement: two half-maps are refined independently and
compared by Fourier shell correlation, so that noise cannot reinforce itself across iterations. The well-known
"Einstein from noise" cautionary example shows how iterative alignment to a reference can manufacture a convincing
signal from pure noise: the failure mode that a held-out criterion exists to catch.

**Analytical cautions.**
1. *Split by similarity, not at random.* Variants one mutation apart are near-duplicates. If close relatives sit on
   both sides of a random train/validation split, validation error looks better than it will be on genuinely new
   designs. Grouping the split by sequence cluster or variant family is the protein-ML analogue of an honest
   R-free set.
2. *Stop on validation, report on test.* Choose the round with validation data, then report performance on variants
   that never influenced any choice, ideally ones characterized later.
3. *Treat residual structure as a clue.* Persistent residuals in one subset, for example variants from one cryo-EM
   dataset processed differently, can reveal a batch effect or a missing descriptor, just as missing height shapes
   the lung-function errors here.
"""
