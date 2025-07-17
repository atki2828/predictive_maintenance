intro = """
### A Statistical Approach Using Survival Regression      
In the realm of practical **predictive maintenance** use cases, it's tempting for data teams to jump straight into **deep learning models**. But recently, I’ve been working with the **Cox Proportional Hazards model with time-varying covariates**, a model rooted in **Survival Analysis**, from the `lifelines` python package, and have discovered it can be an excellent choice on any predictive maintenance use case that entails incorporating telemetry **IoT** signals. It’s interpretable, efficient, and importantly, already familiar to many reliability and quality engineers.

This `streamlit` app explores the Azure Predictive Maintenance dataset to show how a **Cox PH** model can be developed and deployed in a real-world scenario to support smarter maintenance decisions.

I will demonstrate how to extend the **Cox PH** approach with time-varying covariates to model the **probability of a component failure within the next 2 days** as a sample use case. This example illustrates how maintenance teams can leverage these tools to prioritize their work more effectively based on dynamic risk estimates and incorporating ever changing **telemetry data**.

s
Before diving in, here’s why this modeling approach is so well-suited for predictive maintenance:

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


theory_intro = r"""## Theoretical Background🧠  

Survival analysis has its roots in the life sciences, think clinical trials and patient follow-ups, but like all good ideas, it didn’t stay in its lane. Over time, these **“Survival Analysis”** methods found their way into fields like manufacturing, finance, marketing, and of course, predictive maintenance. In broader terms, this family of methods is often called **“Time To Event Analysis.”** If you’re in reliability engineering, you might hear it dressed up as **“Reliability Analysis.”** Same idea, different buzzword.

**Survival Functions**   
Whatever label you slap on it, the core question remains the same: **How long until something important happens?**

In this framework, Itrack a subject say, a machine component, from a known start time, and Ieither observe it fail (mission accomplished) or... nothing happens. That’s fine too. One of the quirks (and strengths) of time to event analysis is the concept of censoring  when the event hasn’t happened yet, but Istill know how long it took to **NOT** happen. That’s useful information! It's like saying: “I didn’t catch fire today.” Good to know.

Mathematically, the survival function $𝑆(𝑡)$  gives us the probability that the event of interest hasn’t occurred by time **t:** $$S(t)=P(T>t)=1−F(t)$$


**But wait there’s more: the Hazard Function.**

The hazard function $ℎ(𝑡)$ is like the survival function’s edgy sibling. While $𝑆(𝑡)$ tells you how likely something is to survive past time **𝑡** the hazard function tells you the instantaneous risk of failure at exactly time **𝑡** given that the subject has made it that far without dying (or breaking).

Formally, the hazard function is defined as:

$$
h(t) = \frac{f(t)}{S(t)}
$$

Or, using the derivative of the survival function:

$$
h(t) = -\frac{d}{dt} \log S(t)
$$

Where:  
-  $f(t)$ is the probability density function (PDF) of the event times  
-  $S(t)$ is the survival function

Think of  $h(t)$ as the **failure intensity**: how risky it is to still be alive (or operating) at time $t$.

For this notebook, we’ll focus on machine components. Some fail and get swapped out **(Event of Interest)**, while others are replaced preemptively during routine maintenance **(Censored Observations)**.

So imagine we’ve got a cohort of machines in an industrial setting. Ilet them run and track how long they last before failing, up to 45 days. Some don’t make it, others are still kicking by the end of the observation window. Below, we’ve got two plots to visualize this:

On the left, a lollipop plot showing each machine’s "lifetime" (with ✖️ for failure and 🟠 for censoring).

In the middle, a fitted Weibull Survival Curve

On the right, a fitted Weibull Hazard Curve"""


theory_cox_models = r"""
**Cox Proportional Hazards Models**  

When it comes to survival modeling, the Cox Proportional Hazards (CPH) model is the workhorse. It’s semi-parametric which is a fancy way of saying it doesn’t assume a specific baseline distribution like the Weibull, but instead focuses on how covariates (a.k.a. features) affect the hazard rate.

The model expresses the hazard for a subject at time *t* as:

$$
h(t) = h_0(t) \cdot \exp(\beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip})
$$

Or more compactly:

$$
h(t) = h_0(t) \cdot \exp(\mathbf{x}_i^\top \boldsymbol{\beta})
$$

Where:  
- $h_0(t)$ is the **baseline hazard** (the risk when all covariates are zero)  
- $\boldsymbol{\beta}$ are the model coefficients  
- $\mathbf{x}_i$ are the covariates for subject *i*

This formulation lets us compare how different factors, like machine vibration, voltage, pressure scale the hazard. A positive coefficient means increasing that feature boosts the hazard (bad news), while a negative one suggests it's protective (your new favorite sensor reading).

One key assumption here is the **proportional hazards assumption**: the ratio of hazards between any two cohort members is **constant over time**. In other words, if Component A is twice as risky as Component B today, it stays twice as risky tomorrow and forever more.

For **inference**, this assumption can matter. And sometimes it can matter a lot! If you're making decisions or writing papers and this assumption doesn't hold, your hazard ratios might lie to you. But for **prediction**? Not a deal-breaker. If you're just trying to forecast failure probabilities and support operational decisions, you can often get away with it especially if you use **probability calibration techniques** down the line (e.g., Platt scaling, isotonic regression).

**Note**: This notebook will not cover calibration as it is long enough already

---

**Cox Models with Time-Varying Covariates**  

Now let’s make things a bit more realistic. Out in the field, maintenance technicians, data teams, and reliability engineers often have deep knowledge of how telemetry signals,things like **increasing vibration**, or **voltage drift** can serve as early warning signs of impending failures. These patterns don’t stay fixed over time, and neither should our models

The basic form is the same, but the covariates are allowed to evolve over time:

$$
h(t) = h_0(t) \cdot \exp(\beta_1 x_{1}(t) + \beta_2 x_{2}(t) + \dots + \beta_p x_{p}(t))
$$

Same interpretability: coefficients still tell you the effect of a covariate on the hazard, but now **at a specific time**. So if component pressure is high at time *t*, and the coefficient for pressure is positive, then the model says “yeah, the risk is climbing *right now*.”

It still assumes proportional hazards, but now in the context of **instantaneous covariate values**. And again for **inference**, violations are a red flag. But for **predictive maintenance**, it’s usually fine. Our end goal is to rank machines by risk, not to get peer-reviewed.
"""
