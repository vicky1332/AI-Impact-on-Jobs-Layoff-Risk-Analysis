
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Impact on Jobs & Layoff Risk", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("ai-impact-jobs-layoff-risk-dataset.csv")

try:
    df = load_data()
except Exception:
    st.error("Place 'ai-impact-jobs-layoff-risk-dataset.csv' in the same folder as app.py")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "Demographic Analysis",
        "Industry Analysis",
        "AI & Automation",
        "Layoff Risk",
        "Correlation",
        "Conclusion",
    ],
)

if page == "Home":
    st.title("🤖 AI Impact on Jobs & Layoff Risk Analysis")
    st.markdown("""
    ### Problem Statement
    This project analyzes how AI adoption, automation exposure, workforce characteristics,
    and employee skills influence layoff risk.
    """)
    c1,c2,c3=st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Industries", df["Industry"].nunique())

elif page=="Dataset":
    st.title("Dataset Overview")
    st.dataframe(df.head())
    st.write(df.describe(include="all").T)

elif page=="Demographic Analysis":
    st.title("Demographic Analysis")
    tabs=st.tabs(["Age","Education","Experience"])
    with tabs[0]:
        fig,ax=plt.subplots()
        sns.histplot(df["Age"],bins=20,kde=True,ax=ax)
        st.pyplot(fig)
    with tabs[1]:
        fig,ax=plt.subplots(figsize=(7,4))
        sns.countplot(data=df,x="Education_Level",order=df["Education_Level"].value_counts().index,ax=ax)
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with tabs[2]:
        fig,ax=plt.subplots()
        sns.histplot(df["Years_of_Experience"],bins=20,kde=True,ax=ax)
        st.pyplot(fig)

elif page=="Industry Analysis":
    st.title("Industry Analysis")
    metric=st.selectbox("Metric",["AI_Adoption_Level","Tasks_Automated_Percentage","Routine_Task_Percentage"])
    g=df.groupby("Industry")[metric].mean().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(10,5))
    sns.barplot(x=g.values,y=g.index,ax=ax)
    ax.set_xlabel(metric)
    st.pyplot(fig)

elif page=="AI & Automation":
    st.title("AI & Automation")
    if "Automation_Exposure_Score" in df.columns:
        fig,ax=plt.subplots()
        sns.boxplot(data=df,x="Layoff_Risk",y="Automation_Exposure_Score",ax=ax)
        st.pyplot(fig)
    if "Work_Adaptability_Score" in df.columns:
        fig,ax=plt.subplots()
        sns.boxplot(data=df,x="Layoff_Risk",y="Work_Adaptability_Score",ax=ax)
        st.pyplot(fig)

elif page=="Layoff Risk":
    st.title("Layoff Risk Analysis")
    fig,ax=plt.subplots()
    sns.countplot(data=df,x="Layoff_Risk",order=["Low","Medium","High"],ax=ax)
    st.pyplot(fig)

    numcols=[c for c in ["Routine_Task_Percentage","Tasks_Automated_Percentage","AI_Training_Hours"] if c in df.columns]
    if numcols:
        feat=st.selectbox("Compare feature",numcols)
        fig,ax=plt.subplots()
        sns.barplot(data=df,x="Layoff_Risk",y=feat,errorbar=None,ax=ax)
        st.pyplot(fig)

elif page=="Correlation":
    st.title("Correlation Analysis")
    tmp=df.copy()
    if "Layoff_Risk" in tmp.columns:
        tmp["Layoff_Risk_Num"]=tmp["Layoff_Risk"].map({"Low":1,"Medium":2,"High":3})
    corr=tmp.select_dtypes("number").corr(method="spearman")
    fig,ax=plt.subplots(figsize=(10,8))
    sns.heatmap(corr,cmap="coolwarm",center=0,ax=ax)
    st.pyplot(fig)

elif page=="Conclusion":
    st.title("Key Findings")
    st.markdown("""
- Higher automation exposure is associated with higher layoff risk.
- Creativity and human interaction are associated with lower automation exposure.
- Workforce adaptability is linked to greater employment resilience.
- AI adoption and automation vary across industries.
- Upskilling and AI training remain important for workforce readiness.
""")
