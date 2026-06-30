# ==========================================================
# AI IMPACT ON JOBS & LAYOFF RISK DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Impact on Jobs & Layoff Risk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------

st.markdown("""
<style>

.main{
    background:#f8fafc;
}

h1{
    color:#1E3A8A;
}

h2{
    color:#2563EB;
}

h3{
    color:#0F766E;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #E5E7EB;
    box-shadow:0px 2px 8px rgba(0,0,0,.08);
}

.stAlert{
    border-radius:12px;
}

</style>
""",unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():

    df=pd.read_csv("ai-impact-jobs-layoff-risk-dataset.csv")

    return df

df=load_data()

# ----------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------

df["Automation_Exposure_Score"]=(
    df["Routine_Task_Percentage"]+
    df["Tasks_Automated_Percentage"]
)/2

df["AI_Engagement_Score"]=(
    df["Number_of_AI_Tools_Used"]*
    df["AI_Usage_Hours_Per_Week"]
)

df["Work_Adaptability_Score"]=(
    df["Creativity_Requirement"]+
    df["Human_Interaction_Level"]+
    df["AI_Training_Hours"]
)/3

df["AI_Adoption_Level_Num"] = df["AI_Adoption_Level"].map({
    "Low": 1.0,
    "Medium": 2.0,
    "High": 3.0
})

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("🤖 Navigation")

page=st.sidebar.radio(

    "",

    [

        "🏠 Home",

        "📊 Dataset Overview",

        "👥 Workforce Analysis",

        "🤖 AI & Automation",

        "📈 Research Questions",

        "📌 Key Findings",

        "📝 Conclusion"

    ]

)

st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Global Filters")

industry=st.sidebar.selectbox(

    "Industry",

    ["All"]+sorted(df["Industry"].unique())

)

company=st.sidebar.selectbox(

    "Company Size",

    ["All"]+sorted(df["Company_Size"].unique())

)

risk=st.sidebar.selectbox(

    "Layoff Risk",

    ["All","Low","Medium","High"]

)

# ----------------------------------------------------------
# FILTERED DATAFRAME
# ----------------------------------------------------------

filtered_df=df.copy()

if industry!="All":

    filtered_df=filtered_df[
        filtered_df["Industry"]==industry
    ]

if company!="All":

    filtered_df=filtered_df[
        filtered_df["Company_Size"]==company
    ]

if risk!="All":

    filtered_df=filtered_df[
        filtered_df["Layoff_Risk"]==risk
    ]

# ----------------------------------------------------------
# CORRELATION MATRIX (GLOBAL)
# ----------------------------------------------------------
corr_df = filtered_df.copy()
corr_df["Layoff_Risk_Num"] = corr_df["Layoff_Risk"].map({
    "Low": 1,
    "Medium": 2,
    "High": 3
})
corr_matrix = corr_df[
    [
        "AI_Engagement_Score",
        "Automation_Exposure_Score",
        "Work_Adaptability_Score",
        "Layoff_Risk_Num"
    ]
].corr(method="spearman")

# ----------------------------------------------------------
# DOWNLOAD BUTTON
# ----------------------------------------------------------

st.sidebar.download_button(

    "⬇ Download Filtered Dataset",

    filtered_df.to_csv(index=False),

    file_name="filtered_dataset.csv",

    mime="text/csv"

)

st.sidebar.markdown("---")

st.sidebar.info("""

### Dashboard

✔ Home

✔ Dataset Overview

✔ Workforce Analysis

✔ AI & Automation

✔ Research Questions

✔ Key Findings

✔ Conclusion

""")

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🤖 AI Impact on Jobs & Layoff Risk Dashboard")

    st.markdown("""
Artificial Intelligence is transforming industries by automating repetitive tasks,
enhancing productivity, and reshaping workforce requirements. While AI creates new
opportunities, it also increases the risk of workforce displacement in jobs that are
highly repetitive and easily automatable.

This interactive dashboard explores how workforce demographics, AI adoption,
automation exposure, employee skills, and organizational characteristics influence
layoff risk using a synthetic dataset of **20,000 employee records**.
""")

    st.markdown("---")

    st.subheader("📊 Dashboard Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Employees", f"{filtered_df.shape[0]:,}")

    with col2:
        st.metric("Industries", filtered_df["Industry"].nunique())

    with col3:
        st.metric("Job Roles", filtered_df["Job_Role"].nunique())

    with col4:
        st.metric(
            "Average AI Adoption",
            round(filtered_df["AI_Adoption_Level_Num"].mean(), 2)
        )

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("🎯 Problem Statement")

        st.write("""
The rapid adoption of Artificial Intelligence has significantly changed workplace
operations across industries. Organizations increasingly automate routine tasks,
which improves efficiency but also introduces uncertainty regarding employment
stability.

The objective of this project is to analyze how AI adoption, workforce
characteristics, automation exposure, and employee adaptability influence
layoff risk and workforce resilience.
""")

    with right:

        st.info("""
### Dataset

- 📂 Source: Kaggle
- 👥 Records: 20,000
- 🏭 Multiple Industries
- 🤖 AI Adoption Metrics
- ⚙ Automation Metrics
- 🎯 Target Variable: Layoff Risk
""")

    st.markdown("---")

    st.subheader("🎯 Project Objectives")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
- Analyze workforce demographics
- Study AI adoption across industries
- Explore automation exposure
- Understand workforce adaptability
- Examine layoff risk patterns
""")

    with c2:

        st.markdown("""
- Identify vulnerable industries
- Discover resilient employee groups
- Answer key business questions
- Generate actionable insights
- Support AI workforce planning
""")

    st.markdown("---")

    st.subheader("📖 About the Dataset")

    st.write("""
The dataset contains **20,000 synthetic employee records** covering multiple
industries, job roles, and organizational structures.

It includes demographic information, professional experience, AI adoption
metrics, automation indicators, employee skill requirements, and a target
variable representing **Layoff Risk (Low, Medium, High)**.

The dataset is well suited for exploratory data analysis, business intelligence,
machine learning, workforce analytics, and AI adoption research.
""")

    st.markdown("---")

    st.subheader("🚀 Dashboard Sections")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success("""
### 👥 Workforce

- Age
- Education
- Experience
- Company Size
- Job Level
""")

    with c2:

        st.success("""
### 🤖 AI Analysis

- AI Adoption
- Automation Exposure
- AI Training
- AI Engagement
- Adaptability
""")

    with c3:

        st.success("""
### 📈 Insights

- Research Questions
- Key Findings
- Business Insights
- Final Conclusion
""")

    st.markdown("---")

    st.info("""
💡 **Tip:** Use the **Global Filters** in the sidebar to analyze specific industries,
company sizes, or layoff risk categories. All charts throughout the dashboard
update automatically based on your selections.
""")

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.markdown("""
This section provides a comprehensive overview of the dataset, including its
structure, statistical summary, missing values, and target variable distribution.
""")

    st.markdown("---")

    # ======================================================
    # Dataset Preview
    # ======================================================

    st.subheader("📄 Dataset Preview")

    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Dataset Shape
    # ======================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows", filtered_df.shape[0])

    with c2:
        st.metric("Columns", filtered_df.shape[1])

    with c3:
        st.metric("Missing Values", int(filtered_df.isnull().sum().sum()))

    st.markdown("---")

    # ======================================================
    # Column Information
    # ======================================================

    st.subheader("📌 Dataset Information")

    info_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": filtered_df.dtypes.astype(str),
        "Missing Values": filtered_df.isnull().sum().values
    })

    st.dataframe(info_df, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Statistical Summary
    # ======================================================

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        filtered_df.describe().T,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # Missing Values
    # ======================================================

    st.subheader("🔍 Missing Value Analysis")

    missing = filtered_df.isnull().sum()

    if missing.sum() == 0:

        st.success("✅ No missing values are present in the dataset.")

    else:

        miss_df = missing[missing > 0].reset_index()

        miss_df.columns = ["Column", "Missing Values"]

        st.dataframe(miss_df, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Target Variable Distribution
    # ======================================================

    st.subheader("🎯 Layoff Risk Distribution")

    col1, col2 = st.columns([2,1])

    with col1:

        fig, ax = plt.subplots(figsize=(8,5))

        sns.countplot(
            data=filtered_df,
            x="Layoff_Risk",
            order=["Low","Medium","High"],
            palette="viridis",
            ax=ax
        )

        ax.set_xlabel("Layoff Risk")
        ax.set_ylabel("Employee Count")

        for p in ax.patches:

            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x()+0.30, p.get_height()+40)
            )

        st.pyplot(fig)

    with col2:

        counts = filtered_df["Layoff_Risk"].value_counts()

        st.metric("Low Risk", counts.get("Low",0))
        st.metric("Medium Risk", counts.get("Medium",0))
        st.metric("High Risk", counts.get("High",0))

    with st.expander("📌 Observation"):

        st.write("""

• The dataset contains a well-balanced distribution across the three layoff risk categories.

• This balanced representation makes the dataset suitable for comparative analysis and predictive modeling.

""")

    st.markdown("---")

    # ======================================================
    # Numerical & Categorical Features
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔢 Numerical Features")

        num_cols = filtered_df.select_dtypes(include=np.number).columns

        st.write(", ".join(num_cols))

    with col2:

        st.subheader("📝 Categorical Features")

        cat_cols = filtered_df.select_dtypes(exclude=np.number).columns

        st.write(", ".join(cat_cols))

    st.markdown("---")

    # ======================================================
    # Dataset Summary
    # ======================================================

    st.success("""

### 📊 Dataset Summary

✔ 20,000 employee records

✔ 15 input features + 1 target variable

✔ Workforce demographics

✔ Job characteristics

✔ AI adoption metrics

✔ Automation indicators

✔ No missing values

✔ Suitable for EDA, Machine Learning, and Business Intelligence

""")

# ==========================================================
# WORKFORCE ANALYSIS
# ==========================================================

elif page == "👥 Workforce Analysis":

    st.title("👥 Workforce Analysis")

    st.markdown("""
This section explores the demographic and professional characteristics of the workforce,
including employee age, education level, work experience, industry, company size,
and job level.
""")

    st.markdown("---")

    # ======================================================
    # KPI CARDS
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Age",
        round(filtered_df["Age"].mean(),1)
    )

    c2.metric(
        "Average Experience",
        round(filtered_df["Years_of_Experience"].mean(),1)
    )

    c3.metric(
        "Industries",
        filtered_df["Industry"].nunique()
    )

    c4.metric(
        "Job Roles",
        filtered_df["Job_Role"].nunique()
    )

    st.markdown("---")

    # ======================================================
    # AGE DISTRIBUTION
    # ======================================================

    st.subheader("📈 Age Distribution")

    fig, ax = plt.subplots(figsize=(9,4))

    sns.histplot(
        data=filtered_df,
        x="Age",
        bins=20,
        kde=True,
        color="#2563EB",
        ax=ax
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Employees")

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Employee ages are fairly evenly distributed between 21 and 60 years.

• The workforce contains employees across different career stages,
providing balanced demographic representation.

• No single age group dominates the dataset.

""")

    st.markdown("---")

    # ======================================================
    # EDUCATION
    # ======================================================

    st.subheader("🎓 Education Level")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.countplot(
        data=filtered_df,
        x="Education_Level",
        order=filtered_df["Education_Level"].value_counts().index,
        palette="viridis",
        ax=ax
    )

    for p in ax.patches:

        ax.annotate(
            int(p.get_height()),
            (p.get_x()+0.20,p.get_height()+100)
        )

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Bachelor's degree holders represent the largest employee group.

• Master's graduates form the second largest workforce segment.

• High School and PhD employees represent comparatively smaller groups.

""")

    st.markdown("---")

    # ======================================================
    # EXPERIENCE
    # ======================================================

    st.subheader("💼 Years of Experience")

    fig, ax = plt.subplots(figsize=(9,4))

    sns.histplot(
        data=filtered_df,
        x="Years_of_Experience",
        bins=20,
        kde=True,
        color="darkorange",
        ax=ax
    )

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Experience follows a positively skewed distribution.

• Most employees possess low to moderate professional experience.

• Highly experienced employees represent a smaller proportion of the workforce.

""")

    st.markdown("---")

    # ======================================================
    # INDUSTRY DISTRIBUTION
    # ======================================================

    st.subheader("🏭 Industry Distribution")

    industry = filtered_df["Industry"].value_counts()

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        x=industry.values,
        y=industry.index,
        palette="rocket",
        ax=ax
    )

    ax.set_xlabel("Employees")

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Employees are distributed across multiple industries.

• The balanced representation allows meaningful comparison of AI adoption,
automation exposure, and layoff risk across business sectors.

""")

    st.markdown("---")

    # ======================================================
    # COMPANY SIZE
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏢 Company Size")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            data=filtered_df,
            x="Company_Size",
            palette="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("📌 Job Level")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            data=filtered_df,
            x="Job_Level",
            palette="Set2",
            ax=ax
        )

        st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Employees are well distributed across Small, Medium, and Large organizations.

• Entry, Mid, and Senior level employees are also reasonably balanced,
supporting comparative workforce analysis.

""")

    st.markdown("---")

    # ======================================================
    # EDUCATION VS LAYOFF RISK
    # ======================================================

    st.subheader("🎯 Education Level vs Layoff Risk")

    edu_risk = pd.crosstab(
        filtered_df["Education_Level"],
        filtered_df["Layoff_Risk"],
        normalize="index"
    ) * 100

    fig, ax = plt.subplots(figsize=(8,5))

    sns.heatmap(
        edu_risk,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        ax=ax
    )

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• High School employees show a relatively larger proportion of High Layoff Risk.

• Employees with Master's and PhD qualifications exhibit greater employment resilience.

• Educational qualification appears to influence workforce stability.

""")

# ==========================================================
# AI & AUTOMATION ANALYSIS
# ==========================================================

elif page == "🤖 AI & Automation":

    st.title("🤖 AI & Automation Analysis")

    st.markdown("""
This section explores how AI adoption, automation exposure, employee creativity,
AI training, and workforce adaptability influence employment stability.
""")

    st.markdown("---")

    # ======================================================
    # KPI CARDS
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Avg AI Adoption",
        round(filtered_df["AI_Adoption_Level_Num"].mean(),2)
    )

    c2.metric(
        "Avg Automation",
        round(filtered_df["Automation_Exposure_Score"].mean(),2)
    )

    c3.metric(
        "Avg AI Training",
        round(filtered_df["AI_Training_Hours"].mean(),2)
    )

    c4.metric(
        "Avg Adaptability",
        round(filtered_df["Work_Adaptability_Score"].mean(),2)
    )

    st.markdown("---")

    # ======================================================
    # AI ADOPTION VS LAYOFF RISK
    # ======================================================

    st.subheader("📈 AI Adoption Level vs Layoff Risk")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.barplot(
        data=filtered_df,
        x="Layoff_Risk",
        y="AI_Adoption_Level_Num",
        estimator=np.mean,
        errorbar=None,
        palette="viridis",
        order=["Low","Medium","High"],
        ax=ax
    )
    ax.set_ylabel("Average AI Adoption Level (1=Low, 2=Medium, 3=High)")

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Average AI adoption increases across Low, Medium and High layoff risk groups.

• Organizations with greater AI integration generally exhibit higher workforce disruption.

""")

    st.markdown("---")

    # ======================================================
    # AUTOMATION EXPOSURE
    # ======================================================

    st.subheader("⚙ Automation Exposure vs Layoff Risk")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.barplot(
        data=filtered_df,
        x="Layoff_Risk",
        y="Automation_Exposure_Score",
        estimator=np.mean,
        errorbar=None,
        palette="inferno",
        order=["Low","Medium","High"],
        ax=ax
    )

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Employees with High Layoff Risk have the highest automation exposure.

• Routine and repetitive work is more susceptible to AI-driven automation.

""")

    st.markdown("---")

    # ======================================================
    # AI TRAINING
    # ======================================================

    st.subheader("🎓 AI Training Hours vs Layoff Risk")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.barplot(
        data=filtered_df,
        x="Layoff_Risk",
        y="AI_Training_Hours",
        estimator=np.mean,
        errorbar=None,
        palette="magma",
        order=["Low","Medium","High"],
        ax=ax
    )

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• High-risk employees receive more AI training on average.

• Increased training may reflect organizational reskilling efforts rather than lower layoff risk.

""")

    st.markdown("---")

    # ======================================================
    # CREATIVITY VS AUTOMATION
    # ======================================================

    st.subheader("🎨 Creativity Requirement vs Automation Exposure")

    corr = filtered_df["Creativity_Requirement"].corr(
        filtered_df["Automation_Exposure_Score"]
    )

    col1, col2 = st.columns([2,1])

    with col1:

        fig, ax = plt.subplots(figsize=(8,5))

        sns.regplot(
            data=filtered_df,
            x="Creativity_Requirement",
            y="Automation_Exposure_Score",
            scatter_kws={"alpha":0.25},
            line_kws={"color":"red"},
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.metric(
            "Correlation",
            round(corr,2)
        )

    with st.expander("📌 Observation"):

        st.write("""

• Creativity shows a negative relationship with automation exposure.

• Creative and problem-solving roles are comparatively less vulnerable to automation.

""")

    st.markdown("---")

    # ======================================================
    # WORK ADAPTABILITY
    # ======================================================

    st.subheader("💡 Workforce Adaptability vs Layoff Risk")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        data=filtered_df,
        x="Layoff_Risk",
        y="Work_Adaptability_Score",
        palette="Set2",
        order=["Low","Medium","High"],
        ax=ax
    )

    st.pyplot(fig)

    with st.expander("📌 Observation"):

        st.write("""

• Employees with stronger adaptability generally belong to the Low Layoff Risk group.

• Adaptability, creativity and communication collectively improve employment resilience.

""")

    st.markdown("---")

    # ======================================================
    # CORRELATION HEATMAP
    # ======================================================

    st.subheader("📊 Correlation Analysis")

    fig, ax = plt.subplots(figsize=(7,6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        ax=ax
    )

    st.pyplot(fig)

    with st.expander("📌 Key Insight"):

        st.write("""

• Automation Exposure Score has the strongest positive association with Layoff Risk.

• Workforce Adaptability has a negative association with Layoff Risk,
indicating that adaptable employees are more resilient.

""")

# ==========================================================
# RESEARCH QUESTIONS
# ==========================================================

elif page == "📈 Research Questions":

    st.title("📈 Business Insights & Research Questions")

    st.markdown("""
This section answers the key research questions explored during the
Exploratory Data Analysis using visualizations and business insights.
""")

    st.markdown("---")

    # ======================================================
    # RQ1
    # ======================================================

    st.subheader("1️⃣ Which industries are most vulnerable to AI-driven layoffs?")

    industry_risk = pd.crosstab(
        filtered_df["Industry"],
        filtered_df["Layoff_Risk"],
        normalize="index"
    ) * 100

    fig, ax = plt.subplots(figsize=(9,5))

    industry_risk["High"].sort_values().plot(
        kind="barh",
        color="crimson",
        ax=ax
    )

    ax.set_xlabel("High Layoff Risk (%)")

    st.pyplot(fig)

    with st.expander("📌 Business Insight"):

        st.write("""

Industries with a larger proportion of employees in the High Layoff Risk
category are comparatively more vulnerable to AI-driven workforce disruption.

""")

    st.markdown("---")

    # ======================================================
    # RQ2
    # ======================================================

    st.subheader("2️⃣ How does AI adoption differ across industries?")

    ai_industry = filtered_df.groupby("Industry")[
        "AI_Adoption_Level_Num"
    ].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9,5))

    sns.barplot(

        x=ai_industry.values,

        y=ai_industry.index,

        palette="viridis",

        ax=ax

    )

    ax.set_xlabel("Average AI Adoption Level")

    st.pyplot(fig)

    with st.expander("📌 Business Insight"):

        st.write("""

AI adoption differs considerably across industries,
indicating varying levels of digital transformation.

""")

    st.markdown("---")

    # ======================================================
    # RQ3
    # ======================================================

    st.subheader("3️⃣ Which employee groups are most resilient?")

    resilient = filtered_df.groupby(
        "Education_Level"
    )["Work_Adaptability_Score"].mean().sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8,4))

    sns.barplot(

        x=resilient.index,

        y=resilient.values,

        palette="Set2",

        ax=ax

    )

    st.pyplot(fig)

    with st.expander("📌 Business Insight"):

        st.write("""

Employees with higher educational qualifications generally demonstrate
greater adaptability and resilience to AI-driven automation.

""")

    st.markdown("---")

    # ======================================================
    # RQ4
    # ======================================================

    st.subheader("4️⃣ What workforce characteristics are associated with High Layoff Risk?")

    summary = filtered_df.groupby(
        "Layoff_Risk"
    )[

        [

            "Automation_Exposure_Score",

            "Work_Adaptability_Score",

            "AI_Training_Hours"

        ]

    ].mean().round(2)

    st.dataframe(

        summary,

        use_container_width=True

    )

    with st.expander("📌 Business Insight"):

        st.write("""

High-risk employees generally exhibit:

• Higher Automation Exposure

• Lower Workforce Adaptability

• Greater Routine Task Percentage

• Lower Creativity Requirement

• Lower Human Interaction

""")

    st.markdown("---")

    # ======================================================
    # RQ5
    # ======================================================

    st.subheader("5️⃣ What factors contribute most to Layoff Risk?")

    corr = corr_matrix["Layoff_Risk_Num"].sort_values(
        ascending=False
    )

    st.dataframe(

        corr.to_frame("Correlation"),

        use_container_width=True

    )

    with st.expander("📌 Business Insight"):

        st.write("""

Automation Exposure Score demonstrates the strongest positive association
with Layoff Risk.

Workforce Adaptability exhibits a negative association,
indicating that adaptable employees are comparatively more resilient.

""")

    st.markdown("---")

    # ======================================================
    # RQ6
    # ======================================================

    st.subheader("6️⃣ How can organizations prepare employees for an AI-driven future?")

    recommendations = pd.DataFrame({

        "Recommendation":[

            "Invest in AI Upskilling",

            "Promote Creativity",

            "Reduce Routine Work",

            "Improve Workforce Adaptability",

            "Encourage Human-centric Skills"

        ],

        "Business Impact":[

            "High",

            "High",

            "Medium",

            "High",

            "High"

        ]

    })

    st.dataframe(

        recommendations,

        use_container_width=True,

        hide_index=True

    )

    with st.expander("📌 Final Insight"):

        st.write("""

Organizations should focus on continuous learning,
AI literacy, creativity, and communication skills to
build a resilient workforce capable of adapting to
future technological changes.

""")

# ==========================================================
# KEY FINDINGS
# ==========================================================

elif page == "📌 Key Findings":

    st.title("📌 Key Findings")

    st.markdown("""
This section summarizes the most important insights obtained from the exploratory
data analysis. These findings highlight the relationship between AI adoption,
automation, workforce characteristics, and layoff risk.
""")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        st.success("""

### 🤖 AI & Automation

✔ AI adoption varies across industries.

✔ Higher AI adoption is associated with higher layoff risk.

✔ Employees performing routine tasks have greater automation exposure.

✔ Automation Exposure Score is the strongest predictor of layoff risk.

""")

    with c2:

        st.success("""

### 👥 Workforce

✔ Creativity reduces automation exposure.

✔ Workforce adaptability improves employment resilience.

✔ Higher educational qualifications are associated with lower layoff risk.

✔ Human interaction supports job stability.

""")

    st.markdown("---")

    st.subheader("📊 Executive Summary")

    summary = pd.DataFrame({

        "Business Area":[

            "AI Adoption",

            "Automation Exposure",

            "Workforce Adaptability",

            "Education",

            "Creativity",

            "Human Interaction"

        ],

        "Impact":[

            "Increases Layoff Risk",

            "Strong Positive Association",

            "Reduces Layoff Risk",

            "Improves Resilience",

            "Reduces Automation",

            "Improves Stability"

        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("💼 Business Recommendations")

    recommendations = [

        "Invest in continuous AI upskilling programs.",

        "Encourage creativity and problem-solving skills.",

        "Reduce dependence on repetitive manual tasks.",

        "Develop workforce adaptability through continuous learning.",

        "Promote communication and collaboration-based roles.",

        "Implement proactive workforce planning before AI deployment."

    ]

    for item in recommendations:

        st.markdown(f"✅ {item}")

    st.markdown("---")

    st.info("""

### Overall Finding

Artificial Intelligence is transforming workplaces by automating routine work.
Employees performing repetitive tasks are more vulnerable to layoffs, whereas
creativity, education, adaptability, and human interaction significantly
improve workforce resilience.

""")

# ==========================================================
# CONCLUSION
# ==========================================================

elif page == "📝 Conclusion":

    st.title("📝 Project Conclusion")

    st.markdown("""
Artificial Intelligence is rapidly transforming modern workplaces by automating
routine tasks, improving operational efficiency, and reshaping workforce
requirements across industries.

This exploratory data analysis demonstrates that employees performing highly
routine and automatable tasks generally face greater layoff risk, whereas
creativity, workforce adaptability, human interaction, and higher educational
qualifications contribute significantly to employment resilience.

Although AI adoption differs across industries and organizations, the findings
suggest that workforce preparedness, continuous learning, and AI upskilling
are critical for reducing employment vulnerability in an AI-driven economy.
""")

    st.markdown("---")

    st.subheader("📊 Project Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Records",
        "20,000"
    )

    c2.metric(
        "Research Questions",
        "6"
    )

    c3.metric(
        "Engineered Features",
        "3"
    )

    c4.metric(
        "Visualizations",
        "12+"
    )

    st.markdown("---")

    st.subheader("🎯 Objectives Achieved")

    st.success("""

✔ Performed comprehensive exploratory data analysis.

✔ Examined workforce demographics.

✔ Analyzed AI adoption and automation exposure.

✔ Identified workforce characteristics associated with layoff risk.

✔ Answered key business research questions.

✔ Generated actionable business insights.

✔ Developed an interactive Streamlit dashboard.

""")

    st.markdown("---")

    st.subheader("💼 Business Recommendations")

    st.markdown("""

- Invest in AI literacy and continuous employee upskilling.

- Promote creativity and innovation-oriented work.

- Encourage collaboration and communication-intensive roles.

- Reduce dependence on repetitive manual tasks.

- Develop workforce adaptability through lifelong learning.

- Use AI as a tool for augmentation rather than replacement wherever possible.

""")

    st.markdown("---")

    st.info("""

### Final Takeaway

Artificial Intelligence is not merely replacing jobs—it is transforming them.
Organizations that invest in employee adaptability, creativity, and continuous
learning will be better positioned to build a resilient workforce capable of
thriving alongside AI technologies.

""")

    st.markdown("---")

    st.caption(
        "Developed by Trivikram Kambhampati | AI Impact on Jobs & Layoff Risk Dashboard | Streamlit"
    )