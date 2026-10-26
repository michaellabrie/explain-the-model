"""The Researcher's Regression Lab: three sections that each explain one fixed regression model.

Run locally:  uv run --with streamlit,pandas,numpy,plotly streamlit run streamlit_app.py
Each section lives in sections/; every number comes from outputs/*.json or the m10 helpers.
"""

import streamlit as st

from sections import boosting, forest, home, linear

st.set_page_config(page_title="The Researcher's Regression Lab", page_icon=":material/biotech:")
sections = [
    st.Page(linear.page, title="Push & Pull", url_path="linear", icon=":material/balance:"),
    st.Page(forest.page, title="Wisdom of the Forest", url_path="forest", icon=":material/forest:"),
    st.Page(boosting.page, title="One Correction at a Time", url_path="boosting", icon=":material/stairs:"),
]
# Streamlit serves the default page at "/" and ignores its url_path, so the overview is the default
# and every section keeps a working /linear, /forest or /boosting link.
overview = st.Page(lambda: home.page(sections), title="Overview", icon=":material/home:", default=True)
st.navigation([overview, *sections]).run()
