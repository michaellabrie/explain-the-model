"""Overview: what the lab is, and what each of the three sections teaches."""

from __future__ import annotations

import streamlit as st

import m10
from sections.common import LABELS

# (model, core question, what a visitor should be able to do afterwards), in navigation order.
SECTIONS = [
    ("Linear regression", "Which features push a prediction up or down, and by how much?",
     "rebuild a prediction from a starting point plus one signed push per feature, and tell a coefficient apart "
     "from one person's push."),
    ("Random forest", "How do many different trees combine into one prediction?",
     "explain how varied trees combine by averaging, and tell disagreement between trees apart from real "
     "uncertainty about a prediction."),
    ("Gradient boosting", "How does each new tree improve the model, and when should training stop?",
     "explain how boosting adds one small correction per round, and use validation error to choose when to stop."),
]


def page(sections: list) -> None:
    st.title("The Researcher's Regression Lab")
    st.markdown("Three regression models, opened up one step at a time. Each section takes one fixed model, a real "
                "person from a health survey, and follows the arithmetic from the model's first number to its final "
                "prediction, with the mathematics shown alongside the charts.")
    for pg, (model, question, goal) in zip(sections, SECTIONS):
        with st.container(border=True):
            st.page_link(pg, label=f"**{pg.title}**: {model}", icon=pg.icon)
            st.markdown(f"*{question}*  \nAfterwards you can {goal}")

    st.markdown("**How to use it.** In any section, pick a dataset and one of five example people in the sidebar. "
                "The example people are real survey participants from the validation rows (no model was trained on "
                "them), and the same five appear in every section, so you can follow one person through all three "
                "models.")
    rows = []
    for key in m10.DATASETS:
        b = m10.load_bundle(key)
        rows.append(f"- **{LABELS[key]}**: {b['target_label']} in {b['units']}, from {b['source']} "
                    f"({b['n_train']} training and {b['n_validation']} validation people).")
    st.markdown("\n".join(rows))
    st.caption("Every number comes from the three fixed models' supplied outputs and helpers; nothing is retrained in "
               "this app. Each section ends with a collapsible Researcher's Perspective that links the method to "
               "structural biology. Those scenarios are hypothetical and contain no fitted models or results.")
