"""Wisdom of the Forest: 100 different trees, one averaged prediction.

The forest's prediction is the average of all 100 tree guesses; averaging fewer is a "partial forest".
The spread of tree guesses is disagreement between trees, never a confidence interval.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sections.common import (column_level, feature, hline, num, outcome, palette, pick_person,
                             researcher_perspective, short_label, show, signed, step, style, table_view, unit, vline)


def page() -> None:
    b, person = pick_person()
    u, rf = b["units"], b["forest"]
    trees = rf["people"][person["id"]]["tree_predictions"]
    n = rf["n_trees"]
    st.title("Wisdom of the Forest")
    st.markdown(f"##### How do {n} different trees combine into one prediction?")
    st.markdown(f"A **random forest** is {n} **decision trees**. Each tree is a flowchart of yes/no questions about "
                f"the features that ends in a guess for **{outcome(b)} ({u})**. Every tree learns from a slightly "
                f"different version of the training data, so the trees disagree. The forest's prediction is simply "
                f"the **average of all {n} guesses**.")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Average of all {n} trees ({u})", num(rf["people"][person["id"]]["prediction"]),
              help="The forest's prediction.")
    c2.metric(f"Tree 1 alone ({u})", num(trees[0]))
    c3.metric(f"Measured ({u})", num(person["observed"]))
    st.markdown(f"For {person['label']}, the {n} trees' guesses run from **{num(min(trees))} to {num(max(trees))} {u}**. "
                "That spread is *disagreement between trees*, not a confidence interval (step 4 shows why).")

    _one_tree(b, person)
    _why_trees_differ(b)
    _averaging(b, person)
    _disagreement(b, person)
    researcher_perspective("resampling, averaging and their limits in structural data", PERSPECTIVE)


# ---- 1. One tree -------------------------------------------------------------
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
    note = " (just above the exact, unrounded cut-off)" if goes == ">" and abs(x - t) < 1e-9 else ""
    return {"#": k, "Question": f"Is {short_label(b, f)} ≤ {t:g}{u}?", "This person": f"{x:g}{u}{note}",
            "Answer": "Yes" if goes == "≤" else "No"}


def _one_tree(b: dict, person: dict) -> None:
    u, rf = b["units"], b["forest"]
    traced = rf["people"][person["id"]]["traced_trees"]
    step(1, "How one tree makes a prediction")
    st.markdown("A tree asks one yes/no question at a time, such as *Is age ≤ 46.5 years?*, and follows the answer "
                "down a branch until no questions are left. That end point is a **leaf**. The leaf's value is the "
                "average outcome of the training rows that landed there when the tree was built, and it is the "
                "tree's entire prediction.")
    names = [f"Tree {i + 1}" for i in range(len(traced))]
    pick = st.segmented_control(f"Follow {person['label']} through one of the first {len(traced)} trees", names,
                                default=names[0], key=f"trace_{b['id']}") or names[0]
    i = names.index(pick)
    t = traced[i]
    rows = [_question(b, s, person["values"], k + 1) for k, s in enumerate(t["steps"])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=38 + 35 * len(rows))
    st.markdown(f"After **{len(t['steps'])} questions**, {person['label']} reaches a leaf worth "
                f"**{num(t['leaf_value'])} {u}**: that is tree {i + 1}'s prediction.")

    depths = [len(tt["steps"]) for tt in traced]
    st.dataframe(pd.DataFrame([{"Tree": f"Tree {j + 1}", "First question": _question(b, tt["steps"][0], person["values"], 1)["Question"],
                                "Questions asked": len(tt["steps"]), f"Leaf value = prediction ({u})": num(tt["leaf_value"])}
                               for j, tt in enumerate(traced)]), hide_index=True)
    st.markdown(f"Same person, five trees: different questions, different leaves, different guesses. These trees "
                f"have no depth limit (scikit-learn's default), so each keeps splitting until its leaves can't be "
                f"split further. That's why the paths here are {min(depths)}–{max(depths)} questions long and a leaf "
                f"often holds just one or a few training rows. Each tree fits its own training sample almost "
                f"perfectly, which on its own makes it a noisy predictor for new people. Trees 6–{rf['n_trees']} "
                f"work the same way.")


# ---- 2. Why the trees differ -------------------------------------------------
def _why_trees_differ(b: dict) -> None:
    c, rf = palette(), b["forest"]
    rows_n = rf["n_training_rows"]
    seen = [100 * k / rows_n for k in rf["bootstrap_unique_training_rows"]]
    expected = 100 * (1 - (1 - 1 / rows_n) ** rows_n)
    step(2, "Why the trees disagree: different rows, different questions")
    st.markdown(f"**(a) Different rows: bootstrap samples.** Each tree is trained on {rows_n} rows drawn **with "
                f"replacement** from the {rows_n} training rows. Some rows are drawn two or more times, others "
                f"never. The chance that a given row is never drawn is")
    st.latex(r"\left(1 - \tfrac{1}{n}\right)^{n} \approx e^{-1} \approx 0.368"
             rf"\quad\Rightarrow\quad \text{{each tree sees about }} {expected:.1f}\% \text{{ of the distinct rows.}}")
    fig = go.Figure()
    fig.add_histogram(x=seen, nbinsx=14, marker=dict(color=c["blue"]), name="Trees",
                      hovertemplate="%{x}% of rows<br>%{y} trees<extra></extra>")
    fig.add_vline(x=expected, line_color=c["muted"], line_width=2,
                  annotation_text=f"Expected: {expected:.1f}%", annotation_position="top")
    style(fig, "Distinct training rows a tree saw (% of all training rows)", "Number of trees", height=300, legend=False)
    fig.update_layout(bargap=0.08, margin=dict(t=40))
    show(fig, key="forest_bootstrap")
    lo, hi = min(rf["bootstrap_unique_training_rows"]), max(rf["bootstrap_unique_training_rows"])
    st.caption(f"Across the {rf['n_trees']} trees, each saw {lo}–{hi} of the {rows_n} training rows "
               f"({min(seen):.1f}–{max(seen):.1f}%, mean {np.mean(seen):.1f}%), close to the expected {expected:.1f}%.")

    k, nf = rf["features_tried_per_split"], rf["n_features"]
    st.markdown(f"**(b) Different questions: random feature choice.** At every split, a tree may choose only among "
                f"**{k} randomly drawn features out of {nf}** (the `sqrt` setting: √{nf} rounded down). The best "
                f"question among those few wins, so different trees, and different splits inside one tree, end up "
                f"asking about different features. Here is each tree's very first question:")
    roots = rf["root_split"]
    counts = Counter(r["feature"] for r in roots)
    order = [f for f, _ in counts.most_common()]
    cuts = {f: sorted({r["threshold"] for r in roots if r["feature"] == f}) for f in order}
    fig = go.Figure()
    fig.add_bar(y=[short_label(b, f) for f in order], x=[counts[f] for f in order], orientation="h",
                marker=dict(color=c["blue"], cornerradius=4), text=[counts[f] for f in order], textposition="outside",
                cliponaxis=False, customdata=[", ".join(f"{t:g}" for t in cuts[f]) for f in order],
                hovertemplate="%{y}: first question in %{x} trees<br>cut-offs used: %{customdata}<extra></extra>")
    style(fig, f"Trees (out of {rf['n_trees']}) whose first question uses this feature", "",
          height=110 + 42 * len(order), legend=False)
    fig.update_yaxes(autorange="reversed")
    show(fig, key="forest_roots")
    top = order[0]
    msg = (f"Even the most common opening feature, **{short_label(b, top)}**, starts only {counts[top]} of the "
           f"{rf['n_trees']} trees, and {len(order)} different features appear as a first question.")
    if len(cuts[top]) > 1:
        u = unit(feature(b, top)["units"])
        msg += (f" Trees that do open with {short_label(b, top)} still pick {len(cuts[top])} different cut-offs "
                f"({cuts[top][0]:g}{u} to {cuts[top][-1]:g}{u}), because each saw different rows.")
    st.markdown(msg)


# ---- 3. Averaging ------------------------------------------------------------
def _averaging(b: dict, person: dict) -> None:
    c, u, rf = palette(), b["units"], b["forest"]
    trees = rf["people"][person["id"]]["tree_predictions"]
    n, forest_pred, obs = rf["n_trees"], rf["people"][person["id"]]["prediction"], person["observed"]
    step(3, "Averaging: many guesses become one prediction")
    k = st.slider("Trees included in the average", 1, n, 20, key="forest_k")
    partial = sum(trees[:k]) / k
    name = (f"Partial forest: average of the first {k} of {n} trees" if k < n
            else f"Whole forest: average of all {n} trees")
    shown = " + ".join(num(t) for t in trees[:min(k, 3)]) + (r" + \cdots" if k > 3 else "")
    st.latex(rf"\hat y_{{\text{{{'partial forest' if k < n else 'forest'}}}}} = \frac{{1}}{{{k}}}\sum_{{b=1}}^{{{k}}} T_b(x)"
             rf" = \frac{{{shown}}}{{{k}}} = {num(partial)}\ \text{{{u}}}")
    st.markdown(f"**{name}: {num(partial)} {u}.** The forest's prediction uses all {n} trees: "
                f"**{num(forest_pred)} {u}**. Measured: {num(obs)} {u}.")

    idx = np.arange(1, n + 1)
    fig = go.Figure()
    if k < n:
        fig.add_scatter(x=trees[k:], y=idx[k:], mode="markers", name="Trees left out",
                        marker=dict(size=8, color=c["faint"]), hovertemplate=f"tree %{{y}}: %{{x:.1f}} {u}<extra></extra>")
    fig.add_scatter(x=trees[:k], y=idx[:k], mode="markers", name=f"Trees in the average ({k})",
                    marker=dict(size=9, color=c["blue"], opacity=0.6, line=dict(width=1.5, color=c["surface"])),
                    hovertemplate=f"tree %{{y}}: %{{x:.1f}} {u}<extra></extra>")
    vline(fig, forest_pred, f"Whole forest ({n} trees): {num(forest_pred)}", c["blue"], 0.5, n + 0.5, 3)
    if k < n:
        vline(fig, partial, f"Partial forest ({k} trees): {num(partial)}", c["orange"], 0.5, n + 0.5, 3)
    vline(fig, obs, f"Measured: {num(obs)}", c["aqua"], 0.5, n + 0.5, 2)
    fig.add_annotation(x=trees[0], y=1, text="Tree 1", showarrow=True, arrowhead=0, ax=36, ay=-18)
    style(fig, f"Each tree's guess for {outcome(b)} ({u})", "Tree number (fixed order)", height=540)
    fig.update_yaxes(range=[n + 2, -1], tickvals=[1, 20, 40, 60, 80, 100])
    show(fig, key="forest_dots")
    st.caption("Each dot is one tree's guess for this person. Move the slider to average more or fewer trees; the "
               "slider always takes trees in the same fixed order, starting from tree 1.")

    run = np.cumsum(trees) / idx
    v = rf["validation_rmse_by_trees"]
    left, right = st.columns(2)
    with left:
        st.markdown("**This person's average as trees are added**")
        fig = go.Figure()
        fig.add_scatter(x=idx, y=run, mode="lines", name="Average of the first k trees", line=dict(color=c["blue"], width=2),
                        hovertemplate=f"first %{{x}} trees: %{{y:.1f}} {u}<extra></extra>")
        hline(fig, obs, "Measured", c["aqua"], 1, n)
        fig.add_scatter(x=[k], y=[partial], mode="markers", name=f"k = {k}",
                        marker=dict(size=11, color=c["orange"], line=dict(width=2, color=c["surface"])))
        show(style(fig, "Number of trees averaged (k)", f"Prediction ({u})", height=320), key="forest_running")
    with right:
        st.markdown("**Validation error of a partial forest**")
        fig = go.Figure()
        fig.add_scatter(x=idx, y=v, mode="lines", name="Validation RMSE", line=dict(color=c["orange"], width=2),
                        hovertemplate=f"first %{{x}} trees: %{{y:.1f}} {u}<extra></extra>")
        fig.add_scatter(x=[k], y=[v[k - 1]], mode="markers", name=f"k = {k}: {num(v[k - 1])} {u}",
                        marker=dict(size=11, color=c["orange"], line=dict(width=2, color=c["surface"])))
        show(style(fig, "Number of trees averaged (k)", f"Validation RMSE ({u})", height=320), key="forest_rmse")

    m1, m2, m3 = st.columns(3)
    m1.metric("Tree 1 alone: validation RMSE", f"{num(v[0])} {u}")
    m2.metric(f"All {n} trees: validation RMSE", f"{num(rf['rmse']['validation'])} {u}")
    m3.metric(f"All {n} trees: training RMSE", f"{num(rf['rmse']['train'])} {u}")
    st.markdown(f"Validation RMSE is the typical error on the {b['n_validation']} held-out people. Averaging cuts it "
                f"from {num(v[0])} {u} (one tree) to {num(rf['rmse']['validation'])} {u} (all {n}), falling fast at "
                f"first and then flattening. The forest's much lower training error ({num(rf['rmse']['train'])} {u}) "
                f"shows the trees still partly memorize their training rows.")

    st.markdown("**Why averaging helps, and why it stops helping.** Think of one tree's guess as random (random "
                "because of its bootstrap rows and feature draws), with variance σ², and let ρ be the correlation "
                "between two trees' guesses. The variance of an average of B trees is")
    st.latex(r"\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b(x)\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2")
    st.markdown("Adding trees shrinks only the second term. The first, ρσ², stays however many trees you add: "
                "whatever all trees have in common doesn't average away. That's why the forest deliberately makes "
                "its trees different (bootstrap rows and random features lower ρ), and one reason the validation "
                "curve flattens instead of falling forever.")


# ---- 4. Disagreement vs uncertainty --------------------------------------------
def _disagreement(b: dict, person: dict) -> None:
    c, u, rf = palette(), b["units"], b["forest"]
    n = rf["n_trees"]
    step(4, "Disagreement between trees is not uncertainty about the answer")
    st.markdown("The spread of the tree guesses tells you how much trees trained on different resamples of the same "
                "data **disagree**. It is tempting to read it as an error bar. It isn't one. Compare the five example "
                "people:")
    stats = []
    for p in b["people"]:
        t = np.array(rf["people"][p["id"]]["tree_predictions"])
        stats.append(dict(p=p, t=t, lo=t.min(), hi=t.max(), sd=t.std(), pred=rf["people"][p["id"]]["prediction"],
                          obs=p["observed"]))

    fig = go.Figure()
    labels = [f"<b>{s['p']['label']}</b>" if s["p"]["id"] == person["id"] else s["p"]["label"] for s in stats]
    jitter = ((np.arange(n) % 9) - 4) / 26
    for i, s in enumerate(stats):
        fig.add_scatter(x=s["t"], y=i + jitter, mode="markers", name=f"Tree guesses ({n} per person)",
                        legendgroup="trees", showlegend=i == 0, marker=dict(size=6, color=c["blue"], opacity=0.35),
                        hovertemplate=f"{s['p']['label']}<br>a tree's guess: %{{x:.1f}} {u}<extra></extra>")
    fig.add_scatter(x=[s["pred"] for s in stats], y=list(range(len(stats))), mode="markers",
                    name="Forest prediction (average)", marker=dict(symbol="line-ns", size=26, line=dict(width=4, color=c["blue"])),
                    hovertemplate=f"forest prediction: %{{x:.1f}} {u}<extra></extra>")
    fig.add_scatter(x=[s["obs"] for s in stats], y=list(range(len(stats))), mode="markers", name="Measured",
                    marker=dict(symbol="diamond", size=14, color=c["aqua"], line=dict(width=2, color=c["surface"])),
                    hovertemplate=f"measured: %{{x:.1f}} {u}<extra></extra>")
    style(fig, f"{outcome(b)} ({u})", "", height=430)
    fig.update_yaxes(tickvals=list(range(len(stats))),
                     ticktext=[f"{lab}<br><sub>{s['p']['why_chosen']}</sub>" for lab, s in zip(labels, stats)],
                     autorange="reversed")
    show(fig, key="forest_people")

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
        points.append("Here every example person's measured value happens to fall inside the trees' range, but "
                      "nothing guarantees that: the range comes with no stated coverage.")
    points.append(f"**No tree can go beyond its training data.** Every leaf averages real training outcomes, so no "
                  f"tree, and so no forest, can predict below {num(lo_t)} or above {num(hi_t)} {u}."
                  + (f" {', '.join(s['p']['label'] for s in edge)}: tree guesses reach that very edge." if edge else ""))
    st.markdown("\n".join(f"- {pt}" for pt in points))
    table_view(pd.DataFrame([{"Person": s["p"]["label"], "Picked as": s["p"]["why_chosen"],
                              f"Lowest tree ({u})": num(s["lo"]), f"Highest tree ({u})": num(s["hi"]),
                              f"Tree SD ({u})": num(s["sd"]), f"Forest ({u})": num(s["pred"]),
                              f"Measured ({u})": num(s["obs"]),
                              "Measured inside tree range?": "yes" if s["lo"] <= s["obs"] <= s["hi"] else "no"}
                             for s in stats]))
    st.info(f"A **confidence or prediction interval** answers *how far could the true value plausibly be from this "
            f"prediction?*, with a stated coverage such as 95%. The tree spread answers a different question. All "
            f"{n} trees learned from the same {rf['n_training_rows']} rows and the same features, so they share blind "
            f"spots (the ρσ² term above) and can agree while all being wrong. Its width also depends on model "
            f"settings such as tree depth and features tried per split, not just on the data. Call it "
            f"**disagreement**, never a confidence interval.", icon=":material/info:")


PERSPECTIVE = """
**Bootstrap, the same idea you meet in phylogenetics.** Bootstrap support values on a phylogenetic tree come from
resampling alignment columns with replacement and asking how often a clade reappears. A random forest resamples
training rows in the same way, but for a different purpose: to make each tree *different*, so that their average
is more stable than any single tree.

**Averaging, the same idea as particle averaging in cryo-EM.** Averaging thousands of noisy particle images raises
the signal-to-noise ratio because the noise in each image is independent. Averaging trees cancels the quirks each
tree picked up from its own bootstrap sample in the same way. Anything shared by *all* the inputs survives the
average, though. In cryo-EM, a systematic problem such as preferred particle orientation isn't fixed by collecting
more particles from the same biased distribution. In a forest, whatever every tree has in common (the same training
rows, the same missing feature) isn't fixed by adding trees. That is the ρσ² term.

**A hypothetical structural-bioinformatics use.** Suppose each row were an AAV capsid variant, the target a
continuous property such as thermal melting temperature (Tm, °C) or an interface binding free energy (ΔG,
kcal/mol), and the features were structure-derived descriptors: buried surface area at the 3-fold interface,
hydrogen-bond and salt-bridge counts, local resolution of the cryo-EM map around the mutation. A forest could
pick up thresholds and interactions without being told about them, for example an effect that appears only above
some buried area, or only at one interface type. Reading its output would need the same discipline as this page:

- The spread of 100 tree guesses for a new variant is **disagreement among trees**, not an error bar on its Tm.
  Calibrated uncertainty would need something else: replicate measurements, residuals on held-out variants,
  conformal prediction, or quantile regression forests.
- A forest **cannot extrapolate**. A designed variant more stable than anything in the training set would be
  predicted no higher than the training maximum. That matters when the whole point of a screen is to find the
  extremes.
- Trees split on whichever features are informative *in this data set*. Feature-use counts like the first-question
  chart above say where the model looked, not which residues matter physically.
"""
