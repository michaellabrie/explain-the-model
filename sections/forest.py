"""Module 2 (Wisdom of the Forest): 100 different trees, one averaged prediction.

The forest's prediction is the average of all 100 tree guesses; averaging fewer is a "partial forest".
The spread of tree guesses is disagreement between trees, never a confidence interval.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sections.common import (column_level, feature, feature_profile, fig_caption, figure, hline, note, num, outcome,
                             page_header, palette, pick_person, research_translation, section, short_label, show,
                             signed, style, subject_record, subsection, table, table_view, unit, vline)


def page() -> None:
    b, person = pick_person()
    u, rf = b["units"], b["forest"]
    trees = rf["people"][person["id"]]["tree_predictions"]
    n, k, nf = rf["n_trees"], rf["features_tried_per_split"], rf["n_features"]
    page_header(b, 2, "Wisdom of the Forest", "Random forest", "Bootstrap Ensemble Averaging in a Random Forest",
                f"How do {n} different trees combine into one prediction?",
                f"Random forest regressor · {n} trees · bootstrap samples · {k} of {nf} features tried per split "
                "(max_features = sqrt) · no depth limit")

    section(1)
    st.markdown(
        f"**Question.** How does a random forest turn {n} different decision-tree guesses into one prediction of "
        f"{outcome(b, True)} ({u}), and what does the spread of those guesses mean?\n\n"
        f"**Learning objective.** Explain how varied trees combine by averaging, and tell *disagreement between trees* "
        f"apart from real *uncertainty about a prediction*.\n\n"
        f"**Approach.** Follow one subject through individual trees, identify why the trees differ, average them "
        f"progressively, and compare their spread with the measured outcomes of all five validation subjects.")

    section(2, "From one decision tree to an averaged ensemble")
    st.latex(rf"\hat y_{{\text{{forest}}}}(x) = \frac{{1}}{{B}}\sum_{{b=1}}^{{B}} T_b(x), \qquad B = {n}")
    st.markdown(
        f"A **decision tree** is a flowchart of yes/no questions about the features that ends in a guess. The forest "
        f"is built and used in six steps:\n"
        f"1. **Resample.** Draw a *bootstrap sample*: {rf['n_training_rows']} rows drawn with replacement from the "
        f"{rf['n_training_rows']} training rows.\n"
        f"2. **Grow.** Fit a tree to that sample, choosing each split from {k} randomly drawn features out of {nf}, "
        f"and keep splitting until the leaves cannot be split further.\n"
        f"3. **Repeat.** Do this {n} times, giving {n} different trees.\n"
        f"4. **Route.** Send the subject down each tree to a leaf; the leaf's value is that tree's guess.\n"
        f"5. **Average.** The forest's prediction is the mean of the {n} guesses.\n"
        f"6. **Assess.** Compare the spread of the guesses (disagreement) with the measured outcome.")
    st.markdown(
        "| Symbol | Meaning | In this dataset |\n|---|---|---|\n"
        f"| $B$ | Number of trees | {n} |\n"
        "| $T_b(x)$ | Guess of tree $b$ for subject $x$: the value of the leaf the subject reaches | Fig. 1 |\n"
        f"| $n$ | Training rows; each bootstrap sample also has $n$ rows | {rf['n_training_rows']} |\n"
        f"| $m$ | Features each split may choose from | {k} of {nf} |\n"
        "| $\\sigma^2$ | Variance of one tree's guess across possible resamples | §6.3 |\n"
        "| $\\rho$ | Correlation between two trees' guesses | §6.3 |")

    section(3)
    subject_record(b, person)

    section(4, "Feature vector")
    table("1", "Standardized feature profile", feature_profile(b, person["values"]),
          "z = (subject value − training mean) / training SD. Values are the subject's recorded survey values. "
          "Editing feature values is part of Module 1; this module only shows the forest's outputs for the supplied "
          "subjects.")

    section(5, "Ensemble prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Average of all {n} trees ({u})", num(rf["people"][person["id"]]["prediction"]),
              help="The forest's prediction.")
    c2.metric(f"Tree 1 alone ({u})", num(trees[0]))
    c3.metric(f"Measured ({u})", num(person["observed"]))
    st.markdown(f"For {person['label']}, the {n} trees' guesses run from **{num(min(trees))} to {num(max(trees))} {u}**. "
                "That spread is *disagreement between trees*, not a confidence interval (§07).")
    _averaging(b, person)

    section(6, "How individual trees arrive at different answers")
    _one_tree(b, person)
    _why_trees_differ(b)
    _convergence(b, person)

    section(7, "Disagreement is not predictive uncertainty")
    _disagreement(b, person)
    research_translation("resampling, ensemble averaging and calibrated uncertainty",
                         "How bootstrap ensembles relate to resampling and averaging practice in phylogenetics and "
                         "cryo-EM, and what a forest's tree spread can and cannot support in a structural screen.",
                         PERSPECTIVE)


# ---- 05. Ensemble prediction -------------------------------------------------
def _averaging(b: dict, person: dict) -> None:
    c, u, rf = palette(), b["units"], b["forest"]
    trees = rf["people"][person["id"]]["tree_predictions"]
    n, forest_pred, obs = rf["n_trees"], rf["people"][person["id"]]["prediction"], person["observed"]
    with figure("1", "Individual tree guesses and their average",
                f"{person['label']} · one point per tree, in the forest's fixed tree order ({u})"):
        k = st.slider("Trees included in the average", 1, n, 20, key="forest_k")
        partial = sum(trees[:k]) / k
        name = (f"Partial forest: average of the first {k} of {n} trees" if k < n
                else f"Whole forest: average of all {n} trees")
        idx = np.arange(1, n + 1)
        fig = go.Figure()
        if k < n:
            fig.add_scatter(x=trees[k:], y=idx[k:], mode="markers", name="Trees left out",
                            marker=dict(size=7, color=c["faint"]), hovertemplate=f"tree %{{y}}: %{{x:.1f}} {u}<extra></extra>")
        fig.add_scatter(x=trees[:k], y=idx[:k], mode="markers", name=f"Trees in the average ({k})",
                        marker=dict(size=8, color=c["blue"], opacity=0.65, line=dict(width=1.5, color=c["surface"])),
                        hovertemplate=f"tree %{{y}}: %{{x:.1f}} {u}<extra></extra>")
        vline(fig, forest_pred, f"Whole forest ({n} trees): {num(forest_pred)}", c["blue"], 0.5, n + 0.5, 2.5)
        if k < n:
            vline(fig, partial, f"Partial forest ({k} trees): {num(partial)}", c["orange"], 0.5, n + 0.5, 2.5)
        vline(fig, obs, f"Measured: {num(obs)}", c["aqua"], 0.5, n + 0.5, 1.5)
        fig.add_annotation(x=trees[0], y=1, text="Tree 1", showarrow=True, arrowhead=0, ax=36, ay=-18,
                           font=dict(size=11))
        style(fig, f"Tree's guess for {outcome(b, True)} ({u})", "Tree number", height=520)
        fig.update_yaxes(range=[n + 2, -1], tickvals=[1, 20, 40, 60, 80, 100])
        show(fig, key="forest_dots")
        fig_caption("1", "Each point is one tree's guess for this subject. The slider averages the first k trees in the "
                         "forest's fixed order (a partial forest, orange); the forest's actual prediction is the "
                         "average of all trees (blue line). The aqua line is the measured value.")
    shown = " + ".join(num(t) for t in trees[:min(k, 3)]) + (r" + \cdots" if k > 3 else "")
    st.latex(rf"\hat y_{{\text{{{'partial forest' if k < n else 'forest'}}}}} = \frac{{1}}{{{k}}}\sum_{{b=1}}^{{{k}}} T_b(x)"
             rf" = \frac{{{shown}}}{{{k}}} = {num(partial)}\ \text{{{u}}}")
    st.markdown(f"**{name}: {num(partial)} {u}.** The forest's prediction uses all {n} trees: "
                f"**{num(forest_pred)} {u}**. Measured: {num(obs)} {u}.")


# ---- 06. Quantitative interpretation -------------------------------------------
def _question(b: dict, s: dict, values: dict, k: int) -> dict:
    f, t, goes, x = s["feature"], s["threshold"], s["goes"], values[s["feature"]]
    cat = column_level(b, f)
    if cat:
        g, level = cat
        name = level.lower() if g["name"] == "sex" else level
        return {"#": k, "Question": f"Is this person {name}?", "This person": "yes" if x == 1 else "no",
                "Answer": "Yes" if goes == ">" else "No"}
    u = unit(feature(b, f)["units"])
    # Cut-offs are stored rounded; a value equal to the shown cut-off can sit just above the exact one.
    note_text = " (just above the exact, unrounded cut-off)" if goes == ">" and abs(x - t) < 1e-9 else ""
    return {"#": k, "Question": f"Is {short_label(b, f)} ≤ {t:g}{u}?", "This person": f"{x:g}{u}{note_text}",
            "Answer": "Yes" if goes == "≤" else "No"}


def _one_tree(b: dict, person: dict) -> None:
    u, rf = b["units"], b["forest"]
    traced = rf["people"][person["id"]]["traced_trees"]
    subsection("6.1", "Anatomy of a single tree")
    st.markdown("A tree asks one yes/no question at a time, such as *Is age ≤ 46.5 years?*, and follows the answer "
                "down a branch until no questions are left. That end point is a **leaf**. The leaf's value is the "
                "average outcome of the training rows that landed there when the tree was built, and it is the "
                "tree's entire prediction.")
    names = [f"Tree {i + 1}" for i in range(len(traced))]
    pick = st.segmented_control(f"Trace {person['label']} through one of the first {len(traced)} trees", names,
                                default=names[0], key=f"trace_{b['id']}") or names[0]
    i = names.index(pick)
    t = traced[i]
    rows = [_question(b, s, person["values"], k + 1) for k, s in enumerate(t["steps"])]
    table("2", f"Decision path of tree {i + 1}", pd.DataFrame(rows),
          f"After {len(t['steps'])} questions, {person['label']} reaches a leaf worth {num(t['leaf_value'])} {u}: "
          f"tree {i + 1}'s prediction.", height=35 * (len(rows) + 1) + 3)

    depths = [len(tt["steps"]) for tt in traced]
    table("3", f"First {len(traced)} trees for the same subject", pd.DataFrame([
        {"Tree": f"Tree {j + 1}", "First question": _question(b, tt["steps"][0], person["values"], 1)["Question"],
         "Questions asked": len(tt["steps"]), f"Leaf value = prediction ({u})": num(tt["leaf_value"])}
        for j, tt in enumerate(traced)]),
          f"Same subject, different questions, different leaves, different guesses. Trees 6–{rf['n_trees']} work "
          "the same way.")
    st.markdown(f"These trees have no depth limit (scikit-learn's default), so each keeps splitting until its leaves "
                f"cannot be split further. That is why the paths here are {min(depths)}–{max(depths)} questions long "
                f"and a leaf often holds just one or a few training rows. Each tree fits its own training sample "
                f"almost perfectly, which on its own makes it a noisy predictor for new people.")


def _why_trees_differ(b: dict) -> None:
    c, rf = palette(), b["forest"]
    rows_n = rf["n_training_rows"]
    seen = [100 * k / rows_n for k in rf["bootstrap_unique_training_rows"]]
    expected = 100 * (1 - (1 - 1 / rows_n) ** rows_n)
    subsection("6.2", "Sources of variation between trees")
    st.markdown(f"**(a) Different rows: bootstrap samples.** Each tree is trained on {rows_n} rows drawn **with "
                f"replacement** from the {rows_n} training rows. Some rows are drawn two or more times, others "
                f"never. The chance that a given row is never drawn is")
    st.latex(r"\left(1 - \tfrac{1}{n}\right)^{n} \approx e^{-1} \approx 0.368"
             rf"\quad\Rightarrow\quad \text{{each tree sees about }} {expected:.1f}\% \text{{ of the distinct rows.}}")
    lo, hi = min(rf["bootstrap_unique_training_rows"]), max(rf["bootstrap_unique_training_rows"])
    with figure("2", "Bootstrap coverage of the training set",
                f"Distinct training rows seen by each of the {rf['n_trees']} trees (% of {rows_n})"):
        fig = go.Figure()
        fig.add_histogram(x=seen, nbinsx=14, marker=dict(color=c["blue"], line=dict(width=1, color=c["surface"])),
                          name="Trees", hovertemplate="%{x}% of rows<br>%{y} trees<extra></extra>")
        fig.add_vline(x=expected, line_color=c["muted"], line_width=1.5,
                      annotation_text=f"Expected: {expected:.1f}%", annotation_position="top", annotation_font_size=11,
                      annotation_yshift=6)
        style(fig, "Distinct training rows seen (% of all training rows)", "Number of trees", height=300, legend=False)
        fig.update_layout(bargap=0.04, margin=dict(t=44))
        show(fig, key="forest_bootstrap")
        fig_caption("2", f"Each tree saw {lo}–{hi} of the {rows_n} training rows ({min(seen):.1f}–{max(seen):.1f}%, "
                         f"mean {np.mean(seen):.1f}%), close to the theoretical {expected:.1f}%. Every tree therefore "
                         "learns from a different subset of the data.")

    k, nf = rf["features_tried_per_split"], rf["n_features"]
    st.markdown(f"**(b) Different questions: random feature choice.** At every split, a tree may choose only among "
                f"**{k} randomly drawn features out of {nf}** (the `sqrt` setting: √{nf} rounded down). The best "
                f"question among those few wins, so different trees, and different splits inside one tree, end up "
                f"asking about different features.")
    roots = rf["root_split"]
    counts = Counter(r["feature"] for r in roots)
    order = [f for f, _ in counts.most_common()]
    cuts = {f: sorted({r["threshold"] for r in roots if r["feature"] == f}) for f in order}
    with figure("3", "Feature chosen for each tree's first split",
                f"Number of trees (out of {rf['n_trees']}) whose first question uses each feature"):
        fig = go.Figure()
        fig.add_bar(y=[short_label(b, f) for f in order], x=[counts[f] for f in order], orientation="h",
                    marker=dict(color=c["blue"], cornerradius=2), text=[counts[f] for f in order], textposition="outside",
                    cliponaxis=False, customdata=[", ".join(f"{t:g}" for t in cuts[f]) for f in order],
                    hovertemplate="%{y}: first question in %{x} trees<br>cut-offs used: %{customdata}<extra></extra>")
        style(fig, "Number of trees", "", height=110 + 40 * len(order), legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig, key="forest_roots")
        top = order[0]
        msg = (f"Even the most common opening feature, {short_label(b, top)}, starts only {counts[top]} of the "
               f"{rf['n_trees']} trees, and {len(order)} different features appear as a first question.")
        if len(cuts[top]) > 1:
            u = unit(feature(b, top)["units"])
            msg += (f" Trees that do open with {short_label(b, top)} still pick {len(cuts[top])} different cut-offs "
                    f"({cuts[top][0]:g}{u} to {cuts[top][-1]:g}{u}), because each saw different rows.")
        fig_caption("3", msg)


def _convergence(b: dict, person: dict) -> None:
    c, u, rf = palette(), b["units"], b["forest"]
    trees = rf["people"][person["id"]]["tree_predictions"]
    n, obs = rf["n_trees"], person["observed"]
    k = st.session_state.get("forest_k", 20)
    subsection("6.3", "Convergence of the ensemble average")
    idx = np.arange(1, n + 1)
    run = np.cumsum(trees) / idx
    v = rf["validation_rmse_by_trees"]
    with figure("4", "Ensemble size, stability and validation error",
                f"(a) {person['label']}'s average of the first k trees · (b) validation RMSE of the first-k partial forest"):
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_scatter(x=idx, y=run, mode="lines", name="Average of the first k trees",
                            line=dict(color=c["blue"], width=2), hovertemplate=f"first %{{x}} trees: %{{y:.1f}} {u}<extra></extra>")
            hline(fig, obs, "Measured", c["aqua"], 1, n, 1.5)
            fig.add_scatter(x=[k], y=[run[k - 1]], mode="markers", name=f"k = {k}",
                            marker=dict(size=10, color=c["orange"], line=dict(width=2, color=c["surface"])))
            show(style(fig, "Trees averaged (k)", f"(a) Prediction ({u})", height=320), key="forest_running")
        with right:
            fig = go.Figure()
            fig.add_scatter(x=idx, y=v, mode="lines", name="Validation RMSE", line=dict(color=c["orange"], width=2),
                            hovertemplate=f"first %{{x}} trees: %{{y:.1f}} {u}<extra></extra>")
            fig.add_scatter(x=[k], y=[v[k - 1]], mode="markers", name=f"k = {k}: {num(v[k - 1])} {u}",
                            marker=dict(size=10, color=c["orange"], line=dict(width=2, color=c["surface"])))
            show(style(fig, "Trees averaged (k)", f"(b) Validation RMSE ({u})", height=320), key="forest_rmse")
        fig_caption("4", f"(a) The subject's running average settles as trees are added. (b) The typical error on the "
                         f"{b['n_validation']} held-out participants falls quickly and then flattens. The marker "
                         "follows the slider in Fig. 1.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tree 1 alone: validation RMSE", f"{num(v[0])} {u}")
    m2.metric(f"All {n} trees: validation RMSE", f"{num(rf['rmse']['validation'])} {u}")
    m3.metric(f"All {n} trees: training RMSE", f"{num(rf['rmse']['train'])} {u}")
    st.markdown(f"Averaging cuts the validation RMSE from {num(v[0])} {u} (one tree) to "
                f"{num(rf['rmse']['validation'])} {u} (all {n}). The forest's much lower training error "
                f"({num(rf['rmse']['train'])} {u}) shows the trees still partly memorize their training rows.")
    st.markdown("**Why averaging helps, and why it stops helping.** Treat one tree's guess as random (random because "
                "of its bootstrap rows and feature draws), with variance σ², and let ρ be the correlation between two "
                "trees' guesses. The variance of an average of B trees is")
    st.latex(r"\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b(x)\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2")
    st.markdown("Adding trees shrinks only the second term. The first, ρσ², stays however many trees are added: "
                "whatever all trees have in common does not average away. That is why the forest deliberately makes "
                "its trees different (bootstrap rows and random features lower ρ), and one reason the validation "
                "curve in Fig. 4b flattens instead of falling forever.")


# ---- 07. Disagreement vs uncertainty --------------------------------------------
def _disagreement(b: dict, person: dict) -> None:
    c, u, rf = palette(), b["units"], b["forest"]
    n = rf["n_trees"]
    st.markdown("The spread of the tree guesses measures how much trees trained on different resamples of the same "
                "data **disagree**. It is tempting to read it as an error bar. It is not one, as a comparison of the "
                "five validation subjects shows.")
    stats = []
    for p in b["people"]:
        t = np.array(rf["people"][p["id"]]["tree_predictions"])
        stats.append(dict(p=p, t=t, lo=t.min(), hi=t.max(), sd=t.std(), pred=rf["people"][p["id"]]["prediction"],
                          obs=p["observed"]))

    with figure("5", "Tree disagreement against measured outcomes",
                f"All five validation subjects · {n} tree guesses each, forest prediction and measured value ({u})"):
        fig = go.Figure()
        labels = [f"<b>{s['p']['label']}</b>" if s["p"]["id"] == person["id"] else s["p"]["label"] for s in stats]
        jitter = ((np.arange(n) % 9) - 4) / 26
        for i, s in enumerate(stats):
            fig.add_scatter(x=s["t"], y=i + jitter, mode="markers", name=f"Tree guesses ({n} per subject)",
                            legendgroup="trees", showlegend=i == 0, marker=dict(size=5, color=c["blue"], opacity=0.35),
                            hovertemplate=f"{s['p']['label']}<br>a tree's guess: %{{x:.1f}} {u}<extra></extra>")
        fig.add_scatter(x=[s["pred"] for s in stats], y=list(range(len(stats))), mode="markers",
                        name="Forest prediction (average)", marker=dict(symbol="line-ns", size=24, line=dict(width=3, color=c["blue"])),
                        hovertemplate=f"forest prediction: %{{x:.1f}} {u}<extra></extra>")
        fig.add_scatter(x=[s["obs"] for s in stats], y=list(range(len(stats))), mode="markers", name="Measured",
                        marker=dict(symbol="diamond", size=12, color=c["aqua"], line=dict(width=2, color=c["surface"])),
                        hovertemplate=f"measured: %{{x:.1f}} {u}<extra></extra>")
        style(fig, f"{outcome(b)} ({u})", "", height=430)
        fig.update_yaxes(tickvals=list(range(len(stats))), showgrid=False,
                         ticktext=[f"{lab}<br><sub>{s['p']['why_chosen']}</sub>" for lab, s in zip(labels, stats)],
                         autorange="reversed")
        show(fig, key="forest_people")
        fig_caption("5", "Each row is one validation subject (the selected subject in bold). Light points are the "
                         "individual tree guesses, the vertical tick is their average (the forest's prediction) and "
                         "the diamond is the measured value. A wide cloud means the trees disagree; it does not mean "
                         "the measured value is likely to fall inside it.")
        table_view(pd.DataFrame([{"Subject": s["p"]["label"], "Selected as": s["p"]["why_chosen"],
                                  f"Lowest tree ({u})": num(s["lo"]), f"Highest tree ({u})": num(s["hi"]),
                                  f"Tree SD ({u})": num(s["sd"]), f"Forest ({u})": num(s["pred"]),
                                  f"Measured ({u})": num(s["obs"]),
                                  "Measured inside tree range?": "yes" if s["lo"] <= s["obs"] <= s["hi"] else "no"}
                                 for s in stats]))

    agree = min(stats, key=lambda s: s["sd"])
    split = max(stats, key=lambda s: s["sd"])
    outside = [s for s in stats if not s["lo"] <= s["obs"] <= s["hi"]]
    lo_t, hi_t = rf["training_target_range"]
    edge = [s for s in stats if s["lo"] <= lo_t or s["hi"] >= hi_t]
    err = agree["pred"] - agree["obs"]
    points = [f"Trees disagree most for **{split['p']['label']}** (guesses from {num(split['lo'])} to "
              f"{num(split['hi'])} {u}, standard deviation {num(split['sd'])} {u}) and agree most for "
              f"**{agree['p']['label']}** (standard deviation {num(agree['sd'])} {u})."]
    if abs(err) > 0.5 * rf["rmse"]["validation"]:
        points.append(f"**Agreement isn't accuracy.** The trees nearly agree on {agree['p']['label']}, yet the forest "
                      f"predicts {num(agree['pred'])} {u} against a measured {num(agree['obs'])} {u} "
                      f"({signed(err)} {u}).")
    else:
        points.append(f"**Agreement isn't a guarantee.** For {agree['p']['label']} the trees nearly agree and the "
                      f"forest is close ({num(agree['pred'])} vs measured {num(agree['obs'])} {u}), but agreement "
                      "and accuracy happened to coincide; nothing in the spread promised it.")
    for s in outside:
        points.append(f"**The spread can miss the truth entirely.** For {s['p']['label']}, all {n} trees guessed "
                      f"between {num(s['lo'])} and {num(s['hi'])} {u}, yet the measured value is {num(s['obs'])} {u}: "
                      "outside every single tree's guess.")
    if not outside:
        points.append("Here every validation subject's measured value happens to fall inside the trees' range, but "
                      "nothing guarantees that: the range comes with no stated coverage.")
    points.append(f"**No tree can go beyond its training data.** Every leaf averages real training outcomes, so no "
                  f"tree, and so no forest, can predict below {num(lo_t)} or above {num(hi_t)} {u}."
                  + (f" {', '.join(s['p']['label'] for s in edge)}: tree guesses reach that very edge." if edge else ""))
    st.markdown("\n".join(f"- {pt}" for pt in points))
    note("Interpretation", f"A **confidence or prediction interval** answers *how far could the true value plausibly be "
         f"from this prediction?*, with a stated coverage such as 95%. The tree spread answers a different question. "
         f"All {n} trees learned from the same {rf['n_training_rows']} rows and the same features, so they share blind "
         f"spots (the ρσ² term in §6.3) and can agree while all being wrong. The spread's width also depends on model "
         f"settings such as tree depth and features tried per split, not just on the data. Call it "
         f"**disagreement**, never a confidence interval.")


PERSPECTIVE = """
**Problem statement.** Suppose each row were an AAV capsid variant and the target a continuous property such as
thermal melting temperature (Tm, °C) or an interface binding free energy (ΔG, kcal/mol), with structure-derived
descriptors as features: buried surface area at the 3-fold interface, hydrogen-bond and salt-bridge counts, local
resolution of the cryo-EM map around a substitution. A random forest can represent thresholds and interactions
without being told about them, for example an effect that appears only above some buried area, or only at one
interface type.

**Formal correspondence.**

| This module (NHANES) | Structural analogue (hypothetical) |
|---|---|
| Bootstrap sample of training rows | Resampled set of variants; cf. resampling alignment columns for phylogenetic bootstrap support |
| One tree $T_b$ | One model fitted to one resample |
| Average of 100 trees | Ensemble estimate; cf. averaging aligned particle images to raise signal-to-noise |
| $(1-\\rho)\\sigma^2/B$ term | Idiosyncratic error that averaging removes, like independent noise across particles |
| $\\rho\\sigma^2$ term | Error shared by every model, like a systematic bias (e.g. preferred particle orientation) that more particles from the same distribution do not fix |
| Spread of tree guesses | Disagreement between resampled models, not an error bar on the measured Tm |
| Training target range | No prediction outside the range of measured training values |

**Analytical cautions.**
1. *Disagreement is not calibrated uncertainty.* Uncertainty statements with stated coverage need a method built
   for them: conformal prediction on held-out variants, quantile regression forests, or replicate measurements that
   estimate experimental error directly.
2. *No extrapolation.* A designed variant more stable than anything in the training set would be predicted no
   higher than the training maximum. In engineering screens, where the aim is often to find the extremes, this
   bound matters.
3. *Feature use is not physical importance.* How often a descriptor is chosen for a split (Fig. 3) reflects what
   is informative *in this dataset*. Correlated descriptors share credit unpredictably, and none of it identifies
   which residues matter mechanistically.
4. *The bootstrap assumes independent rows.* Variants one mutation apart are near-duplicates. Resampling them as if
   independent overstates diversity between trees; resampling whole variant families (a cluster bootstrap) respects
   the dependence.
"""
