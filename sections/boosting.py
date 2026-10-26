"""One Correction at a Time: gradient boosting adds one small, learning-rate-scaled tree per round.

Round 0 = boosting.init (the training average); staged[m] = staged[m-1] + corrections[m-1].
The corrections already include the learning rate, so they are never multiplied by it again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sections.common import (FOCUS_FEATURE, feature, hline, label_of, load_data, num, numeric_features, outcome,
                             palette, pick_person, researcher_perspective, show, signed, step, style, vline)

VIEWS = {"First 100 rounds": 100, "All 500 rounds": 500}


def page() -> None:
    b, person = pick_person()
    u, gb = b["units"], b["boosting"]
    staged = gb["people"][person["id"]]["staged"]
    m = len(staged) - 1
    st.title("One Correction at a Time")
    st.markdown("##### How does each new tree improve the model, and when should training stop?")
    st.markdown(f"**Gradient boosting** builds its prediction for **{outcome(b)} ({u})** as a running sum. It starts "
                f"from one number for everyone, the training average, then adds {m} small trees one at a time. "
                f"Each new tree is fitted to the errors the model still makes on the **training** rows, and only a "
                f"fraction of its output, set by the **learning rate** $\\nu$ = {gb['learning_rate']}, is added.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Round 0 ({u})", num(staged[0]), help="Round 0 is the average outcome in the training rows.")
    c2.metric(f"After round 1 ({u})", num(staged[1]))
    c3.metric(f"After round {m} ({u})", num(staged[m]))
    c4.metric(f"Measured ({u})", num(person["observed"]))

    _algorithm(b)
    _one_person(b, person)
    _residuals(b)
    _stopping(b)
    _learning_rate(b, person)
    _compare(b)
    researcher_perspective("iterative refinement and honest stopping", PERSPECTIVE)


# ---- 1. The algorithm --------------------------------------------------------
def _algorithm(b: dict) -> None:
    u, gb = b["units"], b["boosting"]
    depth, rounds = gb["settings"]["max_depth"], gb["settings"]["n_estimators"]
    step(1, "The algorithm, one line at a time")
    st.latex(rf"""\begin{{aligned}}
F_0(x) &= \bar y_{{\text{{train}}}} = {num(gb['init'], 2)}\ \text{{{u}}} \\
\text{{for }} m &= 1, \dots, {rounds}: \\
r_i &= y_i - F_{{m-1}}(x_i) && \text{{residuals on the training rows}} \\
h_m &= \text{{a depth-{depth} tree fitted to }} (x_i,\ r_i) \\
F_m(x) &= F_{{m-1}}(x) + \underbrace{{\nu\, h_m(x)}}_{{\text{{correction}}}}, \qquad \nu = {gb['learning_rate']}
\end{{aligned}}""")
    st.markdown(
        f"- **Round 0** gives every person the training average, {num(gb['init'])} {u}: the best single "
        f"constant guess when error is measured as squared error.\n"
        f"- **Residual** $r_i$ = measured − current prediction for training row $i$: what is still unexplained.\n"
        f"- **Each tree** $h_m$ is fitted to those residuals, not to the outcome itself. With depth {depth} it asks "
        f"at most {depth} questions (at most {2 ** depth} leaves), so on its own it is a weak, coarse model.\n"
        f"- **Learning rate** $\\nu$: each tree's output is shrunk to {gb['learning_rate']:.0%} before it is added. The "
        f"corrections shown on this page already include $\\nu$; they are never multiplied by it again.\n"
        f"- **Why \"gradient\"?** For squared-error loss $L = \\tfrac12 (y - F)^2$, the slope of the loss with "
        f"respect to the prediction is $-(y - F)$. The residual is the *negative gradient*, so every tree is a small "
        f"step downhill on the training error, taken in the space of prediction functions.")


# ---- 2. One person, round by round ---------------------------------------------
def _one_person(b: dict, person: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    staged, corr = gb["people"][person["id"]]["staged"], gb["people"][person["id"]]["corrections"]
    m, lr, best = len(corr), gb["learning_rate"], gb["best_validation_round"]
    step(2, f"Watch {person['label']}'s prediction move, one tree at a time")
    r = st.slider("Round", 0, m, 1, key="gb_round")
    if r == 0:
        st.latex(rf"F_0 = \bar y_{{\text{{train}}}} = {num(staged[0], 2)}\ \text{{{u}}}")
        st.markdown("Round 0: no trees yet, so the prediction is the average outcome in the training rows.")
    else:
        cr = corr[r - 1]
        st.latex(rf"F_{{{r}}} = F_{{{r - 1}}} + \nu\,h_{{{r}}}(x) = {num(staged[r - 1], 2)} "
                 rf"{'+' if cr >= 0 else '-'} {abs(cr):.2f} = {num(staged[r], 2)}\ \text{{{u}}}")
        st.markdown(f"Tree {r}'s own output for this person was $h_{{{r}}}(x)$ = {signed(cr / lr, 2)} {u}; the "
                    f"model adds only $\\nu$ × that, a correction of **{signed(cr, 2)} {u}**. (That stored correction "
                    f"already includes $\\nu$ = {lr}.)")
    st.caption(f"{person['label']} is a validation row, so no tree ever saw their residual. The corrections they "
               "receive come from patterns the trees learned on training rows with similar feature values.")

    view = st.radio("Rounds shown", list(VIEWS), horizontal=True, key="gb_view_person")
    hi = min(VIEWS[view], m)
    xs = np.arange(hi + 1)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=staged[:hi + 1], mode="lines", name="Prediction after each round",
                    line=dict(color=c["blue"], width=2), hovertemplate=f"round %{{x}}: %{{y:.1f}} {u}<extra></extra>")
    hline(fig, person["observed"], f"Measured: {num(person['observed'])}", c["aqua"], 0, hi)
    lo_y, hi_y = min(min(staged[:hi + 1]), person["observed"]), max(max(staged[:hi + 1]), person["observed"])
    if best <= hi:
        vline(fig, best, f"Best validation round ({best})", c["muted"], lo_y, hi_y, 1.5)
    if r <= hi:
        fig.add_scatter(x=[r], y=[staged[r]], mode="markers", name=f"Round {r}: {num(staged[r])}",
                        marker=dict(size=12, color=c["ink"], line=dict(width=2, color=c["surface"])))
    show(style(fig, "Round (number of trees added)", f"Predicted {outcome(b)} ({u})", height=360), key="gb_path")

    fig = go.Figure()
    fig.add_bar(x=xs[1:], y=corr[:hi], marker=dict(color=[c["red"] if v > 0 else c["blue"] for v in corr[:hi]]),
                hovertemplate=f"tree %{{x}} added %{{y:+.3f}} {u}<extra></extra>", showlegend=False)
    fig.add_hline(y=0, line_color=c["muted"], line_width=1)
    style(fig, "Round (tree number)", f"Correction added ({u})", height=260, legend=False)
    fig.update_layout(bargap=0.15)
    show(fig, key="gb_corrections")
    early, late = np.mean(np.abs(corr[:10])), np.mean(np.abs(corr[100:]))
    st.markdown(f"Upward corrections are red, downward ones blue, each already scaled by $\\nu$. The average size of a "
                f"correction for this person is **{early:.2f} {u}** in rounds 1–10 and **{late:.3f} {u}** in rounds "
                f"101–{m}. Corrections shrink because the training residuals the trees chase shrink. Together, "
                f"rounds 101–{m} move this person's prediction by {signed(staged[m] - staged[100])} {u}.")


# ---- 3. Residuals --------------------------------------------------------------
def _residuals(b: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    snap = gb["residual_snapshots"]
    step(3, "What each new tree is fitted to: the training residuals")
    st.markdown(f"Each tree is fitted to the residuals (measured − current prediction) on the **training** rows. "
                f"Here are {len(snap['row_ids'])} of those training rows at selected rounds. Any pattern left in the "
                "residuals, such as residuals that are mostly positive over one range of a feature, is exactly what "
                "the next tree will try to capture. When no pattern is left, there is little for new trees to "
                "learn except noise.")
    rounds = snap["rounds"]
    r = st.select_slider("Show residuals after round", rounds, value=rounds[0], key="gb_snap")
    obs = np.array(snap["observed"])
    pred = np.array(snap["predictions"][str(r)])
    res = obs - pred
    lim = 1.08 * np.max(np.abs(obs - np.array(snap["predictions"][str(rounds[0])])))
    rows = load_data(b["id"]).set_index("row_id").loc[snap["row_ids"]]
    numeric = [f["name"] for f in numeric_features(b)]
    name = st.selectbox("Plot residuals against", numeric, index=numeric.index(FOCUS_FEATURE.get(b["id"], numeric[0])),
                        format_func=lambda n: label_of(b, n), key=f"gb_feat_{b['id']}")
    f = feature(b, name)
    x = rows[name].to_numpy()

    tab1, tab2 = st.tabs(["Residuals vs. a feature", "Predicted vs. measured"])
    with tab1:
        frame = pd.DataFrame({"x": x, "res": res})
        binned = frame.groupby(pd.qcut(frame.x, 8, duplicates="drop"), observed=True).mean()
        fig = go.Figure()
        fig.add_hline(y=0, line_color=c["muted"], line_width=1)
        fig.add_scatter(x=x, y=res, mode="markers", name="Training rows", marker=dict(size=7, color=c["blue"], opacity=0.5),
                        hovertemplate=f"{f['label']} %{{x}}<br>residual %{{y:+.1f}} {u}<extra></extra>")
        fig.add_scatter(x=binned.x, y=binned.res, mode="lines+markers", name="Average residual (8 equal-size groups)",
                        line=dict(color=c["violet"], width=2.5), marker=dict(size=8, line=dict(width=2, color=c["surface"])))
        style(fig, f"{f['label']} ({f['units']})", f"Residual after round {r} ({u})", height=400)
        fig.update_yaxes(range=[-lim, lim])
        show(fig, key="gb_resid")
    with tab2:
        lo, hi = float(obs.min()), float(obs.max())
        fig = go.Figure()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (measured = predicted)",
                        line=dict(color=c["muted"], width=1.5))
        fig.add_scatter(x=pred, y=obs, mode="markers", name="Training rows", marker=dict(size=7, color=c["blue"], opacity=0.5),
                        hovertemplate=f"predicted %{{x:.1f}}<br>measured %{{y:.1f}} {u}<extra></extra>")
        style(fig, f"Predicted after round {r} ({u})", f"Measured ({u})", height=400)
        fig.update_xaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        show(fig, key="gb_predobs")
    tr = gb["rmse"]["train"]
    st.markdown(f"Training RMSE after round {r}, over all {b['n_train']} training rows: **{num(tr[r])} {u}**. At round "
                f"0 it is {num(tr[0])} {u}, simply the spread of the outcome itself, because everyone gets the same "
                f"prediction. Move the slider to watch the cloud tighten around zero.")


# ---- 4. When to stop -----------------------------------------------------------
def _stopping(b: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    tr, va = gb["rmse"]["train"], gb["rmse"]["validation"]
    best, best_rmse, m = gb["best_validation_round"], gb["best_validation_rmse"], len(tr) - 1
    step(4, "When to stop: training error vs. validation error")
    st.markdown(f"**Training error** is measured on the rows the trees are fitted to, so it keeps falling: every tree "
                f"is built to reduce it. **Validation error** is measured on the {b['n_validation']} held-out rows "
                f"that no tree was fitted to, and shows whether the corrections still help new people. When training "
                f"error keeps falling but validation error rises, the later trees are fitting quirks of the training "
                f"rows: **overfitting**.")
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
        fig.add_scatter(x=[stop, stop], y=[tr[stop], va[stop]], mode="markers", name=f"Your stop: round {stop}",
                        marker=dict(size=12, color=c["ink"], line=dict(width=2, color=c["surface"])))
    show(style(fig, "Round (number of trees)", f"RMSE ({u})", height=380), key="gb_rmse")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Training RMSE at round {stop}", f"{num(tr[stop])} {u}")
    m2.metric(f"Validation RMSE at round {stop}", f"{num(va[stop])} {u}", delta=f"{signed(va[stop] - best_rmse)} {u} vs best",
              delta_color="inverse")
    m3.metric("Lowest validation RMSE", f"{num(best_rmse)} {u}", delta=f"at round {best}", delta_color="off",
              delta_arrow="off")
    st.markdown(f"The lowest validation RMSE is **{num(best_rmse)} {u} at round {best}**. After that, training RMSE "
                f"keeps falling ({num(tr[best])} → {num(tr[m])} {u} by round {m}) while validation RMSE rises "
                f"({num(va[best])} → {num(va[m])} {u}).")
    st.info(f"Stopping at round {best} is a **validation-based choice**, not a test result. The validation rows were "
            f"used to pick the round, so {num(best_rmse)} {u} is slightly optimistic. An honest estimate of error on "
            f"new people would need a separate test set that never influenced any decision.", icon=":material/info:")


# ---- 5. Learning rate ----------------------------------------------------------
def _learning_rate(b: dict, person: dict) -> None:
    c, u, gb = palette(), b["units"], b["boosting"]
    cmp = gb["comparison_learning_rate_0_02"]
    lr, best, cbest = gb["learning_rate"], gb["best_validation_round"], cmp["best_validation_round"]
    va, cva = gb["rmse"]["validation"], cmp["rmse"]["validation"]
    staged, cstaged = gb["people"][person["id"]]["staged"], cmp["people"][person["id"]]["staged"]
    m = len(va) - 1
    step(5, "Learning rate: smaller steps need more rounds")
    st.markdown(f"The learning rate sets how much of each tree's output is added. Here the main model ($\\nu$ = {lr}) is "
                f"compared with a **separately fitted** model with $\\nu$ = 0.02. The 0.02 curves come from that model, "
                f"not from shrinking the 0.1 curves.")
    xs = np.arange(m + 1)
    left, right = st.columns(2)
    with left:
        st.markdown("**Validation RMSE**")
        fig = go.Figure()
        fig.add_scatter(x=xs, y=va, mode="lines", name=f"learning rate {lr}", line=dict(color=c["orange"], width=2),
                        hovertemplate=f"round %{{x}}: %{{y:.2f}} {u}<extra></extra>")
        fig.add_scatter(x=xs, y=cva, mode="lines", name="learning rate 0.02 (separate model)", line=dict(color=c["violet"], width=2),
                        hovertemplate=f"round %{{x}}: %{{y:.2f}} {u}<extra></extra>")
        fig.add_scatter(x=[best, cbest], y=[va[best], cva[cbest]], mode="markers", name="Lowest point of each",
                        marker=dict(size=10, color=c["muted"], line=dict(width=2, color=c["surface"])))
        show(style(fig, "Round", f"Validation RMSE ({u})", height=340), key="gb_lr_rmse")
    with right:
        st.markdown(f"**{person['label']}'s prediction**")
        fig = go.Figure()
        fig.add_scatter(x=xs, y=staged, mode="lines", name=f"learning rate {lr}", line=dict(color=c["blue"], width=2),
                        hovertemplate=f"round %{{x}}: %{{y:.1f}} {u}<extra></extra>")
        fig.add_scatter(x=xs, y=cstaged, mode="lines", name="learning rate 0.02 (separate model)", line=dict(color=c["violet"], width=2),
                        hovertemplate=f"round %{{x}}: %{{y:.1f}} {u}<extra></extra>")
        hline(fig, person["observed"], "Measured", c["aqua"], 0, m)
        show(style(fig, "Round", f"Predicted {outcome(b)} ({u})", height=340), key="gb_lr_person")
    st.dataframe(pd.DataFrame([
        {"Learning rate": str(lr), f"Lowest validation RMSE ({u})": num(gb["best_validation_rmse"]),
         "At round": best, f"This person then ({u})": num(staged[best]), f"Round {m} ({u})": num(staged[m])},
        {"Learning rate": "0.02 (separate)", f"Lowest validation RMSE ({u})": num(cmp["best_validation_rmse"]),
         "At round": cbest, f"This person then ({u})": num(cstaged[cbest]), f"Round {m} ({u})": num(cstaged[m])}]),
        hide_index=True)
    same = abs(cmp["best_validation_rmse"] - gb["best_validation_rmse"]) < 0.01 * gb["best_validation_rmse"]
    st.markdown(f"With $\\nu$ = 0.02 the validation error bottoms out at round **{cbest}** instead of **{best}**, about "
                f"{cbest / best:.0f}× as many rounds, "
                f"{'reaching essentially the same lowest error' if same else 'with a lowest error of'} "
                f"({num(cmp['best_validation_rmse'])} {u} vs {num(gb['best_validation_rmse'])} {u} for $\\nu$ = {lr}). "
                f"Smaller steps take more rounds to get to the same place. In "
                f"exchange, the error curve is typically flatter near its lowest point, so the result is less "
                f"sensitive to exactly where you stop. Both stopping rounds are validation-based choices.")


def _compare(b: dict) -> None:
    u, lin, rf, gb = b["units"], b["linear"], b["forest"], b["boosting"]
    tr, best = gb["rmse"]["train"], gb["best_validation_round"]
    m = len(tr) - 1
    with st.expander("How do the three models compare on this dataset?", icon=":material/compare_arrows:"):
        fair = {"Linear regression": lin["rmse"]["validation"], f"Random forest ({rf['n_trees']} trees)": rf["rmse"]["validation"],
                f"Gradient boosting ({m} rounds)": gb["rmse"]["validation"][m]}
        st.dataframe(pd.DataFrame([
            {"Model": "Linear regression", f"Training RMSE ({u})": num(lin["rmse"]["train"]),
             f"Validation RMSE ({u})": num(lin["rmse"]["validation"]), "Note": "one straight-line equation"},
            {"Model": f"Random forest ({rf['n_trees']} trees)", f"Training RMSE ({u})": num(rf["rmse"]["train"]),
             f"Validation RMSE ({u})": num(rf["rmse"]["validation"]), "Note": "average of fully grown trees"},
            {"Model": f"Gradient boosting ({m} rounds)", f"Training RMSE ({u})": num(tr[m]),
             f"Validation RMSE ({u})": num(gb["rmse"]["validation"][m]), "Note": "the full fitted model"},
            {"Model": f"Gradient boosting stopped at round {best}", f"Training RMSE ({u})": num(tr[best]),
             f"Validation RMSE ({u})": num(gb["best_validation_rmse"]),
             "Note": "round chosen with these same validation rows, so optimistic"}]), hide_index=True)
        winner = min(fair, key=fair.get)
        st.markdown(f"Among the three fixed models as fitted, the lowest validation RMSE here belongs to **{winner}**, "
                    f"at {num(fair[winner])} {u}. When the flexible tree models don't beat a straight line, the "
                    "relationship is close to linear, or the features carry little extra signal. When they beat it "
                    "clearly, there is non-linear structure the line misses. The gap between training and validation "
                    "columns shows how much better each model fits its own training rows than new people. Compare "
                    "within this dataset only, since each dataset has its own units.")


PERSPECTIVE = """
**R-work and R-free.** In crystallographic refinement, the model is refined against most of the reflections
(R-work), while a small held-out set is never used for refinement (R-free). R-work can be driven down almost
indefinitely by adding parameters; a rising gap between R-work and R-free signals that the model is fitting noise.
Training RMSE plays the role of R-work and validation RMSE the role of R-free. Choosing the stopping round from the
validation curve is the same discipline.

**Gold-standard refinement in cryo-EM.** Refining two half-maps independently and comparing them by Fourier shell
correlation keeps noise from reinforcing itself during iterative refinement. The well-known "Einstein from noise"
cautionary example shows how iterative alignment to a reference can manufacture a convincing signal from pure noise.
Boosting is also iterative refinement: each round corrects what is left over. Held-out data is what tells you
whether later rounds still correct signal or have started to chase noise.

**Learning rate as step size.** Small, damped steps in an iterative optimizer need more iterations but are less
likely to overshoot; the same trade-off appears in the $\\nu$ = 0.1 vs 0.02 comparison above.

**A hypothetical structural-bioinformatics use.** Imagine predicting a continuous property of AAV capsid variants,
such as assembly yield or an interface binding free energy (ΔG, kcal/mol), from structure-derived features at the
3-fold and 5-fold interfaces. Boosting would start every variant at the training mean. Early trees would correct
the largest, broadest errors (perhaps a whole class of interfaces); later trees would make smaller, more specific
adjustments. Using it responsibly would mean:

- **Split by similarity, not at random.** Variants one mutation apart are near-duplicates. If close relatives sit
  on both sides of a random train/validation split, validation error looks better than it will be on genuinely new
  designs. Grouping by sequence cluster or variant family is the protein-ML analogue of a held-out R-free set.
- **Stop on validation, report on test.** Pick the round with validation data, then report performance on variants
  that never influenced any choice, ideally ones characterized later.
- **Treat residual structure as a clue.** Persistent residuals in one subset, for example variants from one cryo-EM
  dataset processed differently, can reveal a batch effect or a missing feature, just as missing height shapes the
  lung-function errors here.
"""
