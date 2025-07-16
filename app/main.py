import streamlit as st

def main():
    st.set_page_config(
        page_title="Predictive Maintenance Model Homepage",
        page_icon=🤖,
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.title("Predictive Maintenance with COX PH (TimeVarying Covariates Lifelines)")
    gif_url = "https://media1.tenor.com/m/CfzqzAf4Cf0AAAAd/spiteful-machine-virtualdream.gif"
    st.image(gif_url, caption="Spiteful Machine GIF", use_container_width=True)

if __name__ == "__main__":
    main()