intro = """
### A Statistical Approach Using Survival Regression      
In the realm of practical **predictive maintenance** use cases, it's tempting for data teams to jump straight into **deep learning models**. But recently, I’ve been working with the **Cox Proportional Hazards model with time-varying covariates**, a model rooted in **Survival Analysis**, from the `lifelines` python package, and have discovered it can be an excellent choice on any predictive maintenance use case that entails incorporating telemetry **IoT** signals. It’s interpretable, efficient, and importantly, already familiar to many reliability and quality engineers.

This `streamlit` app explores the Azure Predictive Maintenance dataset to show how a **Cox PH** model can be developed and deployed in a real-world scenario to support smarter maintenance decisions.

I will demonstrate how to extend the **Cox PH** approach with time-varying covariates to model the **probability of a component failure within the next 2 days** as a sample use case. This example illustrates how maintenance teams can leverage these tools to prioritize their work more effectively based on dynamic risk estimates and incorporating ever changing **telemetry data**.

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
    - Weibull Survival and Hazard Distributions

     ---

    ### 📈 Page 3: **Telemetry Feature Engineering Analysis**
    Dive into the data with:
    - Telemetry Time Series
    - Visualizations Voltage, Rotation, Pressure, Vibrations
    - Analysis of Telemetry signal interactions with Machine Errors, Regular Maintenance and **Component Failures** 

    ---

    ### ⚙️ Page 4: **Production Dashboard**
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

In this framework, you track a subject say, a machine component, from a known start time, and either observe it fail (mission accomplished) or... nothing happens. That’s fine too. One of the quirks (and strengths) of time to event analysis is the concept of censoring  when the event hasn’t happened yet, but you still know how long it took to **NOT** happen. That’s useful information! It's like saying: “I didn’t catch fire today.” Good to know.

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

For this demonstration, I will focus on machine components. Some fail and get swapped out **(Event of Interest)**, while others are replaced preemptively during routine maintenance **(Censored Observations)**.

So imagine there is a cohort of machines in an industrial setting. And we let them run and track how long they last before failing, up to 45 days. Some don’t make it, others are still kicking by the end of the observation window. Below, we’ve got two plots to visualize this:

On the left, a lollipop plot showing each machine’s "lifetime" (with ❌ for failure and 🔵 for censoring).

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


eda_intro = r"""## Exploratory Data Analysis 🔎

To demonstrate the practical utility of the Cox Proportional Hazards model for predictive maintenance, we'll be working with the **Microsoft Azure Predictive Maintenance dataset** — a synthetic yet well-structured dataset that simulates machine health over the course of a year across **100 unique machines**.

The available data includes:

1. **Telemetry data**: Hourly readings of Voltage, Rotation, Pressure, and Vibration  
2. **Errors**: Timestamped logs of error codes (types 1–5)  
3. **Maintenance records**: Logs of component replacements (components 1–4), regardless of failure  
4. **Failure records**: Logs of component replacements **due to failure**  
5. **Machine metadata**: Machine age and model type  

> 📌 **Note**: For simplicity, all data has been aggregated to the **daily level** for this analysis.

---

The goal of this **EDA** is to identify **early warning signals** of component failure that can serve as covariates in a **Cox Proportional Hazards (Cox PH)** model. We'll explore patterns and trends in telemetry data, error codes, and maintenance history to better understand the **failure dynamics** and prepare the data for survival modeling."""

component_failure_text = """### Component Failures Across Machines 

Since the predictive maintenance model Iare aiming to build is focused on component failures, let's look at the proportions of failures across the 4 components across all machines.   

Below Ican see Component 2 fails the most frequently while Component 3 fails the least. """


all_machine_failures_text = """### Failures Across All Machines

Looking at the total number of failures across all machines Isee that machine **99** has the most failures with 18."""

time_between_failure_text = """ ### Pure Time Between Failures

Since our model will take into account the time to event, it is worthwhile to take a look at the distribution of time between failures across components."""

time_between_comp_fail_text = """### Time Between Component Failures
Here Ibreakout the distribution of time between component failures across all the 4 components on each machine.

Ican also see that the failure records occur in multiples of 15 days. (product of a contrived likely simulated dataset). """

compare_time_between_fail_text = """ ### Comparing time between regularly scheduled maintenace and component failure

This comparison shows that regular maintenance seems to require replacing the component every 15 days. For the instance marked 0 (in red) Ican see that most of the time components are replaced 15 days and that component failures occur when this regular maintenace was missed. """

telemetry_eda_intro = """
## Telemetry Data

After looking at much of the telemetry data, I found that in this contrived likely A.I. generated dataset there were particular patterns of failure that can be seen across each component. The following plots will detail each one and they will ultimately inform the modeling strategy. 

In the plots below the solid vertical lines are on the day of a component failure, the dashed lines are on the day of a scheduled maintenance and the dotted lines represent the day an error was tripped.
"""

weibull_survival_text = """## Fitting Weibull Survival and Hazard Model

The curves below show the **Survival** and **Hazard** functions for each of the 4 model types. This does not take into account components, but rather models the time to failure between any components on a given machine.

It is evident that both **Model 1** and **Model 2** machines have less reliability (less time between failures) when compared to **Model 3** and **Model 4**."""

telemetry_component_one_text = """
### Component 1 Failure Signatures

A common pattern shown for a component 1 failure is that there is a spike in the **Voltage** and **Error 1** is tripped the previous day. This persists throughout the dataset and can be seen in the example below.

**Machine ID 79**


"""


telemetry_component_two_text = """### Component 2 Failure Signatures

A common pattern shown for a component 2 failure is that there is a dip in the **Rotation** and **Error 2 and Error 3** are tripped the previous day. This persists throughout the dataset and can be seen in the example below.

**Machine ID 23**"""

telemetry_component_three_text = """
### Component 3 Failure Signatures

A common pattern shown for a component 3 failure is that there is a spike in the **Pressure** and **Error 4** is tripped the previous day. This persists throughout the dataset and can be seen in the example below.

**Machine ID 42** 😉
"""


telemetry_component_four_text = """
### Component 4 Failure Signatures

A common pattern shown for a **component 4 failure** is that there is a spike in the **Vibration** and **Error 5** is tripped the previous day. This persists throughout the dataset and can be seen in the example below.

**Machine ID 51**"""


feature_text = """## Features To Be Used For COX PH Model

From the analysis of the telemetry data, it was discovered that each component prior to failure tends to show a large **Spike** or **Dip** in a specific telemetry value around 2 days before the component is replaced due to failure.  

To capture these deviations more robustly, I will compute **21-day rolling means and rolling standard deviations** for each telemetry variable and then calculate a **z-score** for the current day's reading relative to the past 21 days. If the **z-score** on a given day is over a certain threshold... say 2.5, that day will be flagged as being an anomalous telemetry measurement and used as a time varying predictor in the model.

Using these anomaly flags, I will train **4 COX PH models** (one for each component) with:
- **Fixed variables**: Machine model type and machine age  
- **Time-varying predictor**: 21-day rolling z-score anomaly flag which will be a binary 0 or 1

Below are the components and their associated telemetry signals:

- **Component 1** → Spike in Voltage  
- **Component 2** → Dip in Rotation  
- **Component 3** → Spike in Pressure  
- **Component 4** → Spike in Vibration


**The z-score calculation is performed in the cell below**"""
