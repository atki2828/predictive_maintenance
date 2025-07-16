intro = """
### A Statistical Approach Using Survival Regression      
In the realm of practical **predictive maintenance** use cases, it's tempting for data teams to jump straight into **deep learning models**. But recently, I’ve been working with the **Cox Proportional Hazards model with time-varying covariates** from the `lifelines` python package, and have discovered it can be an excellent choice on any predictive maintenance use case that entails incorporating telemetry **IoT** signals. It’s interpretable, efficient, and importantly, already familiar to many reliability and quality engineers.

This `streamlit` app explores that alternative using the Azure Predictive Maintenance dataset to show how a **Cox PH** model can be developed and deployed in a real-world scenario to support smarter maintenance decisions.

Before diving in, here’s why I think this modeling approach is so well-suited for predictive maintenance:

✅ Combines static attributes (e.g., manufacturer, install date) with real-time telemetry from sensors   
✅ Easily integrated into production pipelines using Python and open-source tools like lifelines   
✅ Can yield failure probabilities with a bit of post-processing   
✅ Naturally handles class imbalance and censored data, both common in real-world maintenance settings   
✅ Bridges the gap between data science teams and reliability engineers who already use these models in tools like JMP or Minitab as Lifelines now scalable and automatable in Python
"""

app_overview = """
    ## 📑 App Overview

    This application demonstrates a **Predictive Maintenance** workflow using statistical and machine learning techniques. Use the sidebar to navigate between the following sections:

    ---

    ### 🔍 Page 1: **Theoretical Background**
    Learn about the statistical foundations behind the model, including:
    - **Survival Analysis**
    - **Cox Proportional Hazards Model**
    - **Time-Varying Covariates**

    ---

    ### 📊 Page 2: **Exploratory Data Analysis**
    Dive into the data with:
    - Summary statistics
    - Visualizations of key features
    - Failure event distributions

    ---

    ### ⚙️ Page 3: **Production Dashboard**
    Simulate and view model outputs as they would appear in a production setting.  
    Includes:
    - **Failure risk scores**
    - **Priority rankings**
    - Example real-time KPIs

    ---

    #### 📚 Want More Detail?
    For a deeper dive into **model building, feature engineering, and statistical interpretation**, check out my Kaggle notebook here:  
    [**Predictive Maintenance: A Statistical Approach**](https://www.kaggle.com/code/eigenvalue42/predictive-maintenance-a-statistical-approach)
    """
