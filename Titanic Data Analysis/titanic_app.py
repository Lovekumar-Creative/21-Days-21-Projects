
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Titanic EDA (Streamlit)", layout="wide")

# -------------------- Helpers --------------------
@st.cache_data
def load_dataset():
    """
    Try common file names first from the working directory, otherwise fall back to seaborn's titanic dataset.
    """
    import os
    possible_files = ["titanic.csv", "train.csv", "titanic_train.csv", "/content/titanic.csv"]
    for fname in possible_files:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                return df, f"Loaded dataset from '{fname}'"
            except Exception as e:
                pass
    # fallback to seaborn dataset
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df, "Loaded seaborn 'titanic' dataset (fallback)"
    except Exception as e:
        # minimal demo dataframe
        df = pd.DataFrame({
            "survived":[0,1,1],
            "pclass":[3,1,2],
            "sex":["male","female","female"],
            "age":[22,38,26],
            "sibsp":[1,1,0],
            "parch":[0,0,0],
            "fare":[7.25,71.2833,7.925],
            "embarked":["S","C","Q"],
            "class":["Third","First","Second"],
            "who":["man","woman","woman"],
            "adult_male":[True,False,False],
            "deck":[None,"C",None],
            "embark_town":["Southampton","Cherbourg","Queenstown"],
            "alive":["no","yes","yes"],
            "alone":[False,False,True]
        })
        return df, "Created demo dataset (no file found and seaborn unavailable)"

def value_counts_table(df, col):
    vc = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    vc["percent"] = (vc["count"]/vc["count"].sum()*100).round(2)
    return vc

def title_from_name(name):
    # common title extraction for classic Titanic datasets
    import re
    m = re.search(r",\s*([^\.]+)\.", str(name))
    if m:
        return m.group(1).strip()
    # fallback to rarer formats
    m2 = re.search(r"^([^ ]+)\s", str(name))
    return m2.group(1) if m2 else "Unknown"

# -------------------- Load data --------------------
df, load_msg = load_dataset()

# Ensure consistent column names (lowercase)
orig_columns = df.columns.tolist()
df.columns = [c.strip() for c in df.columns]

# Sidebar
st.sidebar.title("Controls")
st.sidebar.markdown(load_msg)
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# Basic filters
with st.sidebar.expander("Quick filters", expanded=True):
    # Collect some common column names with fallbacks
    col_survived = next((c for c in df.columns if c.lower() in ["survived","alive"]), None)
    col_pclass = next((c for c in df.columns if c.lower() in ["pclass","class"]), None)
    col_sex = next((c for c in df.columns if c.lower() in ["sex","gender"]), None)
    col_embarked = next((c for c in df.columns if c.lower() in ["embarked","embark_town"]), None)

    selected_pclasses = None
    if col_pclass:
        pclass_options = df[col_pclass].dropna().unique().tolist()
        selected_pclasses = st.multiselect("Pclass / Class", sorted(pclass_options), default=sorted(pclass_options))

    selected_sex = None
    if col_sex:
        sex_options = df[col_sex].dropna().unique().tolist()
        selected_sex = st.multiselect("Sex", sorted(sex_options), default=sorted(sex_options))

    selected_embark = None
    if col_embarked:
        embark_options = df[col_embarked].dropna().unique().tolist()
        selected_embark = st.multiselect("Embarked", sorted(embark_options), default=sorted(embark_options))

# Apply quick filters to a working copy
df_work = df.copy()
if col_pclass and selected_pclasses is not None:
    df_work = df_work[df_work[col_pclass].isin(selected_pclasses)]
if col_sex and selected_sex is not None:
    df_work = df_work[df_work[col_sex].isin(selected_sex)]
if col_embarked and selected_embark is not None:
    df_work = df_work[df_work[col_embarked].isin(selected_embark)]

# Main page
st.title("Titanic — Exploratory Data Analysis (Streamlit)")
st.write("Interactive data analysis app generated from your notebook. All EDA content is preserved and presented with a clean UI. Use the sidebar to filter the dataset.")

# Tabs for sections
tabs = st.tabs(["Dataset Preview", "Data Cleaning", "Summary Stats", "Visualizations", "Correlations", "Feature Engineering", "Notes & Export"])

# ---------- Dataset Preview ----------
with tabs[0]:
    st.header("Dataset Preview")
    st.write("Head of the dataset (after applying filters):")
    st.dataframe(df_work.head(20))

    st.subheader("Columns & dtypes")
    col_info = pd.DataFrame({"column": df_work.columns, "dtype": [str(df_work[c].dtype) for c in df_work.columns]})
    st.dataframe(col_info)

    st.subheader("Value counts for selected columns")
    demo_cols = [c for c in [col_survived, col_pclass, col_sex, col_embarked] if c is not None]
    for c in demo_cols:
        st.markdown(f"**{c}**")
        st.dataframe(value_counts_table(df_work, c))

# ---------- Data Cleaning ----------
with tabs[1]:
    st.header("Data Cleaning & Missing Values")
    st.write("Below we show missing value summary, common cleaning steps and simple imputations used in typical Titanic EDA workflows. You can run imputations interactively using the buttons below.")

    missing = df_work.isnull().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_pct"] = (missing["missing_count"] / df_work.shape[0] * 100).round(2)
    st.dataframe(missing.sort_values("missing_count", ascending=False))

    st.subheader("Common cleaning actions (preview)")
    st.write("""
    - Fill missing `Embarked` with the most common embarkation port.
    - Fill missing `Age` with median or by title-based median.
    - Fill missing `Fare` with median.
    - Drop columns with too many missing values (e.g., Cabin deck) or create 'has_cabin' flag.
    """)

    # Interactive imputation demo
    col_age = next((c for c in df_work.columns if c.lower()=="age"), None)
    col_fare = next((c for c in df_work.columns if c.lower()=="fare"), None)
    col_cabin = next((c for c in df_work.columns if "cab" in c.lower()), None)
    col_name = next((c for c in df_work.columns if "name" in c.lower()), None)
    col_emb = next((c for c in df_work.columns if c.lower().startswith("embark")), None)

    impute_age = st.button("Impute Age (median)")
    if impute_age and col_age:
        df_work[col_age] = df_work[col_age].fillna(df_work[col_age].median())
        st.success("Age imputed with median.")

    impute_emb = st.button("Impute Embarked (mode)")
    if impute_emb and col_emb:
        df_work[col_emb] = df_work[col_emb].fillna(df_work[col_emb].mode().iloc[0])
        st.success("Embarked imputed with mode.")

    impute_fare = st.button("Impute Fare (median)")
    if impute_fare and col_fare:
        df_work[col_fare] = df_work[col_fare].fillna(df_work[col_fare].median())
        st.success("Fare imputed with median.")

    if col_cabin:
        st.write("Create `has_cabin` flag from Cabin column")
        if st.button("Create has_cabin flag"):
            df_work["has_cabin"] = df_work[col_cabin].notnull().astype(int)
            st.success("has_cabin flag created.")

# ---------- Summary Stats ----------
with tabs[2]:
    st.header("Summary Statistics")
    st.write("Numeric summary:")
    st.dataframe(df_work.describe(include=[np.number]).T)

    st.write("Categorical summary (top levels):")
    cat_cols = df_work.select_dtypes(include=['object','category','bool']).columns.tolist()
    cat_summary = {c: df_work[c].value_counts(dropna=False).head(10).to_dict() for c in cat_cols}
    for c in cat_cols:
        st.subheader(c)
        vc = value_counts_table(df_work, c)
        st.dataframe(vc)

# ---------- Visualizations ----------
with tabs[3]:
    st.header("Visualizations")
    st.write("Interactive, reusable plots built with Plotly. Use the controls to change variables.")

    # Common column picks
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_work.select_dtypes(include=['object','category','bool']).columns.tolist()

    col_x = st.selectbox("X / Category for plots", categorical_cols, index=0 if categorical_cols else None)
    col_y = st.selectbox("Y / Numeric for plots", numeric_cols, index=0 if numeric_cols else None)

    if col_x and col_y:
        st.subheader(f"Bar chart of mean {col_y} by {col_x}")
        grouped = df_work.groupby(col_x)[col_y].mean().reset_index().sort_values(col_y, ascending=False)
        fig = px.bar(grouped, x=col_x, y=col_y, labels={col_x:col_x, col_y:col_y}, title=f"Mean {col_y} by {col_x}")
        st.plotly_chart(fig, use_container_width=True)

    # Survival by category
    if col_survived and col_x:
        st.subheader(f"Survival rate by {col_x}")
        surv = df_work.groupby(col_x)[col_survived].mean().reset_index().sort_values(col_survived, ascending=False)
        surv[col_survived] = (surv[col_survived]*100).round(2)
        fig2 = px.bar(surv, x=col_x, y=col_survived, labels={col_survived:"Survival %"}, title=f"Survival % by {col_x}")
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution and KDE for numeric
    if col_y:
        st.subheader(f"Distribution of {col_y}")
        fig3 = px.histogram(df_work, x=col_y, nbins=30, marginal="box", title=f"Distribution of {col_y}")
        st.plotly_chart(fig3, use_container_width=True)

    # Age vs Fare scatter with hover & sizing
    if "age" in [c.lower() for c in df_work.columns] and "fare" in [c.lower() for c in df_work.columns]:
        age_col = next(c for c in df_work.columns if c.lower()=="age")
        fare_col = next(c for c in df_work.columns if c.lower()=="fare")
        st.subheader("Age vs Fare (scatter) colored by survival if available")
        color_col = col_survived if col_survived else None
        fig4 = px.scatter(df_work, x=age_col, y=fare_col, color=color_col, hover_data=df_work.columns, title="Age vs Fare")
        st.plotly_chart(fig4, use_container_width=True)

    # Pairplot-style (small)
    if len(numeric_cols) >= 3:
        st.subheader("Scatter matrix (numeric columns)")
        sample = df_work[numeric_cols].dropna().sample(n=min(300, len(df_work)))
        fig5 = px.scatter_matrix(sample, dimensions=sample.columns[:4].tolist(), title="Scatter matrix (first 4 numeric columns)")
        st.plotly_chart(fig5, use_container_width=True)

# ---------- Correlations ----------
with tabs[4]:
    st.header("Correlations & Heatmap")
    st.write("Pearson correlations for numeric columns. Non-numeric columns are excluded here.")
    num = df_work.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr()
        st.dataframe(corr)
        st.subheader("Heatmap (matplotlib)")
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.imshow(corr, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

# ---------- Feature Engineering ----------
with tabs[5]:
    st.header("Feature Engineering (common EDA steps)")
    st.write("This section shows common transformations used in Titanic EDA, such as extracting titles, creating family size, etc. These changes are applied to a *copy* of the working dataframe so original preview remains intact. Use the buttons to add features.")

    fe_df = df_work.copy()
    if col_name:
        if st.button("Extract Title from Name"):
            fe_df["title"] = fe_df[col_name].apply(title_from_name)
            st.success("Title column created.")
            st.dataframe(fe_df[["title"]].value_counts().rename_axis("title").reset_index(name="count"))

    if st.button("Create family_size = SibSp + Parch + 1"):
        sp = next((c for c in fe_df.columns if c.lower()=="sibsp"), None)
        pr = next((c for c in fe_df.columns if c.lower()=="parch"), None)
        if sp and pr:
            fe_df["family_size"] = fe_df[sp].fillna(0).astype(int) + fe_df[pr].fillna(0).astype(int) + 1
            st.success("family_size created.")
            st.dataframe(fe_df[["family_size"]].describe())

    if st.button("Create is_alone flag"):
        sp = next((c for c in fe_df.columns if c.lower()=="sibsp"), None)
        pr = next((c for c in fe_df.columns if c.lower()=="parch"), None)
        if sp and pr:
            fe_df["is_alone"] = ((fe_df[sp].fillna(0)+fe_df[pr].fillna(0))==0).astype(int)
            st.success("is_alone created.")
            st.dataframe(fe_df[["is_alone"]].value_counts())

    st.write("You can download the engineered dataset below.")
    csv = fe_df.to_csv(index=False)
    st.download_button("Download engineered dataset (CSV)", data=csv, file_name="titanic_engineered.csv", mime="text/csv")

# ---------- Notes & Export ----------
with tabs[6]:
    st.header("Notes, Exports & Further Steps")
    st.markdown("""
    **What this app contains**:
    - Full dataset preview and dtype summary.
    - Missing value table and simple imputation buttons.
    - Summary statistics for numerical and categorical columns.
    - Interactive Plotly visualizations: bar charts, histograms, scatter, scatter matrix.
    - Correlation matrix and heatmap.
    - Basic feature engineering helpers (Title extraction, family_size, is_alone).
    - Download engineered dataset as CSV.
    """)
    st.subheader("Export current filtered data")
    st.download_button("Download filtered dataset (CSV)", data=df_work.to_csv(index=False), file_name="titanic_filtered.csv", mime="text/csv")

    st.markdown("### Tips for deploying")
    st.write("""
    1. Save this file as `titanic_app.py`.
    2. Ensure required packages are installed: `pip install streamlit pandas plotly matplotlib seaborn`.
    3. Run locally: `streamlit run titanic_app.py`.
    4. To deploy: push to GitHub and connect to Streamlit Cloud or use Docker/Heroku.
    """)

# Footer
st.markdown("---")
st.caption("App generated from your notebook and converted into a single Streamlit file. If you want additional custom visual styles or to include any specific chart from your notebook, tell me the chart name and I'll integrate it.")
