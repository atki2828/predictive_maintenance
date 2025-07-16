import streamlit as st

from app.app_text import app_overview, intro


def main():
    st.set_page_config(
        page_title="Predictive Maintenance Model Homepage",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )

    gif_url = (
        "https://media1.tenor.com/m/CfzqzAf4Cf0AAAAd/spiteful-machine-virtualdream.gif"
    )
    st.title("Predictive Maintenance")
    st.image(gif_url, caption="Spiteful Machine GIF", use_container_width=True)
    st.divider()
    st.markdown(intro)

    st.divider()
    st.markdown(app_overview)


if __name__ == "__main__":
    main()
