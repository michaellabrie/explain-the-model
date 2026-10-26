"""Overview: scientific framing of the lab, the study design and the three analysis modules."""

from __future__ import annotations

from html import escape

import streamlit as st

import m10
from sections.common import LABELS, meta_grid, numeric_features, rule, short_label

# (formal title, model, core question, what a visitor should be able to do afterwards), in navigation order.
MODULES = [
    ("Additive Decomposition of a Linear Regression Prediction", "Linear regression",
     "Which features push a prediction up or down, and by how much?",
     "rebuild a prediction from a starting point plus one signed push per feature, and tell a coefficient apart "
     "from one person's push."),
    ("Bootstrap Ensemble Averaging in a Random Forest", "Random forest",
     "How do many different trees combine into one prediction?",
     "explain how varied trees combine by averaging, and tell disagreement between trees apart from real "
     "uncertainty about a prediction."),
    ("Stagewise Residual Correction by Gradient Boosting", "Gradient boosting",
     "How does each new tree improve the model, and when should training stop?",
     "explain how boosting adds one small correction per round, and use validation error to choose when to stop."),
]


def page(sections: list) -> None:
    st.markdown('<div class="lab-eyebrow">Computational notebook · interpretable machine learning</div>',
                unsafe_allow_html=True)
    st.title("The Researcher's Regression Lab", anchor=False)
    st.markdown('<div class="lab-question">Interpretable dissection of linear, ensemble and boosted regression models '
                'on NHANES physiological outcomes</div>', unsafe_allow_html=True)
    meta_grid({"Cohorts": "3 NHANES datasets: fasting glucose, systolic blood pressure, lung function (FEV₁)",
               "Partition": "fixed 80/20 training/validation split per dataset (seed 0)",
               "Models": "3 fixed specifications: linear regression, random forest (100 trees), gradient boosting "
                         "(500 rounds)",
               "Subjects": "5 pre-selected validation participants per dataset, shared by all three modules",
               "Provenance": "every value comes from pre-computed model outputs; nothing is retrained in this app"})

    rule("00", "Abstract")
    st.markdown("Three widely used regression models are examined from the inside. Each module takes one fixed, "
                "pre-fitted model and one held-out survey participant, and follows the computation from the model's "
                "first number to its final prediction: the additive contributions of a linear model, the averaging "
                "of 100 bootstrap-trained decision trees, and the stagewise correction of 500 boosted trees. Every "
                "figure is paired with the mathematics it illustrates and with a note on what the analysis can and "
                "cannot support. Each module closes with a research translation that connects the method to "
                "structure-based prediction problems in capsid biology and cryo-EM.")

    rule("01", "Analysis modules")
    for i, (pg, (title, model, question, goal)) in enumerate(zip(sections, MODULES), 1):
        with st.container(border=True):
            st.markdown(f'<div class="lab-eyebrow">Module {i} · {escape(pg.title)} · {model}</div>',
                        unsafe_allow_html=True)
            st.page_link(pg, label=f"**{title}**", icon=pg.icon)
            st.markdown(f"*{question}*  \n**Objective:** {goal[0].upper() + goal[1:]}")

    rule("02", "Cohorts")
    # A Markdown table (not st.dataframe) so the predictor lists wrap instead of scrolling.
    lines = ["| Dataset | Outcome | Survey | Training / validation | Predictors |", "|---|---|---|---|---|"]
    for key in m10.DATASETS:
        b = m10.load_bundle(key)
        predictors = [short_label(b, f["name"]) for f in numeric_features(b)] + [g["label"] for g in b["categorical_groups"]]
        lines.append(f"| {LABELS[key]} | {b['target_label']} ({b['units']}) | {b['source'].split(' (')[0]} | "
                     f"{b['n_train']} / {b['n_validation']} | {', '.join(predictors)} |")
    with st.container(border=True):
        st.markdown('<div class="fig-head"><span class="num">Table 1</span><span class="title">Datasets analysed'
                    '</span></div>', unsafe_allow_html=True)
        st.markdown("\n".join(lines))
        st.caption("Pick a dataset and one of five validation subjects in the sidebar of any module. The subjects are "
                   "real survey participants from the validation rows (no model was trained on them), and the same five "
                   "appear in every module, so one person can be followed through all three models. These data describe "
                   "this survey sample, not the whole US population.")

    rule("03", "Conventions")
    st.markdown("- **Units** are shown on every number, and are those of the outcome being predicted.\n"
                "- **Residual** = measured − predicted. A positive **push** raises a prediction; a negative push "
                "lowers it.\n"
                "- **Associations, not causes.** Every model describes how outcomes co-vary with features in this "
                "sample, not what would happen if a feature were changed.\n"
                "- **RMSE** (root-mean-square error) is in the outcome's units and is never compared across datasets.\n"
                "- **Validation, not test.** Validation rows are used to judge and to tune (the boosting stopping "
                "round), so a validation-based choice is not an independent test result.\n"
                "- **Disagreement, not uncertainty.** The spread of tree guesses in a forest is never presented as a "
                "confidence or prediction interval.")
    st.caption("Research translations are hypothetical scenarios for interpretation only; they contain no fitted "
               "models, experimental results or numerical predictions.")
