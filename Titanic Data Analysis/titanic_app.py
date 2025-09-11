
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Titanic — Friendly EDA", layout="wide", page_icon="🚢")

# -------------------- Helpers --------------------
@st.cache_data
def load_dataset_from_defaults():
    import os
    possible_files = ["titanic.csv", "train.csv", "titanic_train.csv", "/content/titanic.csv"]
    for fname in possible_files:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                return df, f"Loaded dataset from '{fname}'"
            except Exception:
                pass
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df, "Loaded seaborn 'titanic' dataset (fallback)"
    except Exception:
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
    import re
    m = re.search(r",\s*([^\.]+)\.", str(name))
    if m:
        return m.group(1).strip()
    m2 = re.search(r"^([^ ]+)\s", str(name))
    return m2.group(1) if m2 else "Unknown"

def simple_insights(df, col_survived):
    # Generate a short, plain-language summary of top-level insights
    insights = []
    insights.append(f"Total passengers in the current view: **{len(df):,}**.")
    if col_survived in df.columns:
        surv_rate = df[col_survived].dropna().mean()*100
        insights.append(f"Overall survival rate: **{surv_rate:.1f}%** (calculated from the filtered data).")
    if 'age' in [c.lower() for c in df.columns]:
        age_col = next(c for c in df.columns if c.lower()=='age')
        insights.append(f"Average age: **{df[age_col].dropna().mean():.1f} years**. Missing ages: **{int(df[age_col].isna().sum())}**.")
    if 'fare' in [c.lower() for c in df.columns]:
        fare_col = next(c for c in df.columns if c.lower()=='fare')
        insights.append(f"Average fare paid: **{df[fare_col].dropna().mean():.2f}** (currency as in dataset).")
    # common category
    for col in ['embarked','embark_town','class','pclass']:
        candidates = [c for c in df.columns if c.lower()==col]
        if candidates:
            c = candidates[0]
            top = df[c].mode().iloc[0] if not df[c].mode().empty else "N/A"
            insights.append(f"Most common **{c}**: **{top}**.")
            break
    return insights

# -------------------- Load / Input --------------------
with st.sidebar:
    st.title("Controls")
    uploaded = st.file_uploader("Upload a CSV file (optional)", type=["csv"], help="If you have your own Titanic CSV (train.csv), upload it here.")
    st.markdown("**Mode**: Choose how much detail you want.")
    mode = st.radio("Choose analysis mode", ["Simple (for everyone)", "Advanced (for analysts)"])
    st.markdown("---")
    st.markdown("**Quick tips**: Use the filters below to narrow data. Then explore charts or download results.")

# load data (uploaded takes priority)
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        load_msg = "Loaded dataset from uploaded file."
    except Exception as e:
        st.error("Could not read uploaded file as CSV. Falling back to defaults.")
        df, load_msg = load_dataset_from_defaults()
else:
    df, load_msg = load_dataset_from_defaults()

# tidy column names (strip only)
df.columns = [c.strip() for c in df.columns]

# common column detection
col_survived = next((c for c in df.columns if c.lower() in ["survived","alive"]), None)
col_pclass = next((c for c in df.columns if c.lower() in ["pclass","class"]), None)
col_sex = next((c for c in df.columns if c.lower() in ["sex","gender"]), None)
col_embarked = next((c for c in df.columns if c.lower() in ["embarked","embark_town","embark_town"]), None)
col_name = next((c for c in df.columns if "name" in c.lower()), None)

# Sidebar filters (kept compact for non-tech users)
with st.sidebar.expander("Filters (optional)", expanded=True):
    st.markdown(load_msg)
    st.write(f"Rows: **{df.shape[0]:,}** | Columns: **{df.shape[1]}**")
    selected_pclasses = None
    if col_pclass:
        pclass_options = sorted(df[col_pclass].dropna().unique().tolist())
        selected_pclasses = st.multiselect("Pclass / Class", pclass_options, default=pclass_options, help="Pick class(es) to include")
    selected_sex = None
    if col_sex:
        sex_options = sorted(df[col_sex].dropna().unique().tolist())
        selected_sex = st.multiselect("Sex", sex_options, default=sex_options)
    selected_embark = None
    if col_embarked:
        embark_options = sorted(df[col_embarked].dropna().unique().tolist())
        selected_embark = st.multiselect("Embarked", embark_options, default=embark_options)

# Apply filters
df_work = df.copy()
if col_pclass and selected_pclasses is not None:
    df_work = df_work[df_work[col_pclass].isin(selected_pclasses)]
if col_sex and selected_sex is not None:
    df_work = df_work[df_work[col_sex].isin(selected_sex)]
if col_embarked and selected_embark is not None:
    df_work = df_work[df_work[col_embarked].isin(selected_embark)]

# -------------------- Stylish header --------------------
st.markdown(
    '<style> .title-font {font-size:34px; font-weight:700; letter-spacing:0.5px;} .lead {color: #555; font-size:16px;} .card {background:#ffffff; padding:12px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);} </style>',
    unsafe_allow_html=True)

# header image + title
header_col1, header_col2 = st.columns([1.6, 3])
with header_col1:
    st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80",
             caption="RMS Titanic (historic photo) — dataset inspired by the ship", use_container_width=True)
with header_col2:
    st.markdown('<div class="title-font">Titanic — Easy Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="lead">An interactive, beginner-friendly visualization of the Titanic passenger dataset. No programming required — use filters and buttons to explore.</div>', unsafe_allow_html=True)
    st.markdown("**How to use**: Pick filters in the left panel, then explore tabs below. If you want a simplified view, choose **Simple** mode in Controls.", unsafe_allow_html=True)

st.markdown("---")

# Top-level metrics (easy to read)
m1, m2, m3, m4 = st.columns(4)
total = len(df_work)
m1.metric("Passengers (filtered)", f"{total:,}")
if col_survived in df_work.columns:
    surv_pct = df_work[col_survived].dropna().mean()*100 if df_work[col_survived].dropna().size>0 else 0
    m2.metric("Survival rate", f"{surv_pct:.1f}%")
else:
    m2.metric("Survival rate", "N/A")
if 'age' in [c.lower() for c in df_work.columns]:
    age_col = next(c for c in df_work.columns if c.lower()=='age')
    m3.metric("Average age", f"{df_work[age_col].dropna().mean():.1f}" if df_work[age_col].dropna().size>0 else "N/A")
else:
    m3.metric("Average age", "N/A")
if 'fare' in [c.lower() for c in df_work.columns]:
    fare_col = next(c for c in df_work.columns if c.lower()=='fare')
    m4.metric("Average fare", f"{df_work[fare_col].dropna().mean():.2f}" if df_work[fare_col].dropna().size>0 else "N/A")

# Show simple insights for non-tech users
with st.expander('''Show simple insights (plain language)''', expanded=(mode=="Simple (for everyone)")):
    insights = simple_insights(df_work, col_survived) if len(df_work)>0 else ["No data to summarize."]
    for line in insights:
        st.markdown(f"- {line}")

# -------------------- Tabs (core content) --------------------
tabs = st.tabs(["Overview", "Data & Cleaning", "Charts (Simple)", "Charts (Advanced)", "Explore & Download", "About"])

# ---------- Overview ----------
with tabs[0]:
    st.header("Overview — What this dataset is about")
    st.markdown('''
    **Short story for non-technical users:** The Titanic dataset contains information about passengers who were on the RMS Titanic.
    Each row is a passenger and columns describe features like age, sex, passenger class, fare paid, and whether the passenger survived.
    ''')
    st.markdown("**This app helps you:**")
    st.markdown("- See the data (table & column types)\n- Understand missing values and simple fixes\n- Visualize patterns (who survived, fares, ages)\n- Create simple derived columns like family size")
    st.markdown("If you'd like the app to **explain a chart** in plain words, click the small 'Explain chart' checkboxes near each chart.")

    st.subheader("Quick data snapshot (top 10 rows)")
    st.dataframe(df_work.head(10))

    st.subheader("Column types & counts")
    col_info = pd.DataFrame({"column": df_work.columns, "dtype": [str(df_work[c].dtype) for c in df_work.columns]})
    st.dataframe(col_info)

# ---------- Data & Cleaning ----------
with tabs[1]:
    st.header("Data Cleaning & Missing Values (easy)")
    st.write("We show how many values are missing and provide simple one-click fixes. These actions are applied to the **current filtered view**. You can always download the modified dataset.")
    missing = df_work.isnull().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_pct"] = (missing["missing_count"] / df_work.shape[0] * 100).round(2)
    st.dataframe(missing.sort_values("missing_count", ascending=False))

    st.markdown("**One-click fixes** (use if you don't know what to do):")
    col_age = next((c for c in df_work.columns if c.lower()=="age"), None)
    col_fare = next((c for c in df_work.columns if c.lower()=="fare"), None)
    col_cabin = next((c for c in df_work.columns if "cab" in c.lower()), None)
    col_emb = next((c for c in df_work.columns if c.lower().startswith("embark")), None)

    if st.button("Fill missing Age with median"):
        if col_age:
            df_work[col_age] = df_work[col_age].fillna(df_work[col_age].median())
            st.success("Filled missing ages with median.")
            st.rerun()

    if st.button("Fill missing Fare with median"):
        if col_fare:
            df_work[col_fare] = df_work[col_fare].fillna(df_work[col_fare].median())
            st.success("Filled missing fares with median.")
            st.rerun()

    if st.button("Fill missing Embarked with mode"):
        if col_emb:
            df_work[col_emb] = df_work[col_emb].fillna(df_work[col_emb].mode().iloc[0])
            st.success("Filled missing Embarked with most common value.")
            st.rerun()

    if col_cabin:
        if st.button("Create has_cabin flag"):
            df_work["has_cabin"] = df_work[col_cabin].notnull().astype(int)
            st.success("Created has_cabin flag.")
            st.rerun()

# ---------- Charts (Simple) ----------
with tabs[2]:
    st.header("Charts — Beginner-friendly (simple explanations)")
    st.markdown("Below are three simple charts with short explanations you can understand without technical knowledge.")

    # Chart 1: Survival by Sex
    if col_survived and col_sex:
        st.subheader("Survival by Sex")
        df_plot = df_work[[col_sex, col_survived]].dropna()
        surv_gender = df_plot.groupby(col_sex)[col_survived].mean().reset_index()
        surv_gender[col_survived] = (surv_gender[col_survived]*100).round(1)
        fig = px.bar(surv_gender, x=col_sex, y=col_survived, labels={col_survived:"Survival %"}, title="Survival % by Sex")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**What this shows:** The bar height is the percent of passengers in that group who survived. Taller bars mean more survivors in that group.")

    # Chart 2: Age distribution
    age_col = next((c for c in df_work.columns if c.lower()=="age"), None)
    if age_col:
        st.subheader("Age distribution (who was onboard?)")
        fig2 = px.histogram(df_work, x=age_col, nbins=30, title="Age distribution")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**What this shows:** This tells you how many passengers were in each age group. Peaks mean many passengers around that age.")

    # Chart 3: Survivors by Class
    if col_survived and col_pclass:
        st.subheader("Survival by Passenger Class")
        df_pc = df_work[[col_pclass, col_survived]].dropna()
        surv_pc = df_pc.groupby(col_pclass)[col_survived].mean().reset_index()
        surv_pc[col_survived] = (surv_pc[col_survived]*100).round(1)
        fig3 = px.bar(surv_pc, x=col_pclass, y=col_survived, labels={col_survived:"Survival %"}, title="Survival % by Class")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**What this shows:** Compare survival percentages across travel classes.")

# ---------- Charts (Advanced) ----------
with tabs[3]:
    st.header("Charts — Advanced / Exploratory")
    st.write("This area contains the full interactive plotting controls similar to the original notebook. Use it if you want more control.")

    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_work.select_dtypes(include=['object','category','bool']).columns.tolist()

    col_x = st.selectbox("X / Category for plots", categorical_cols, index=0 if categorical_cols else None)
    col_y = st.selectbox("Y / Numeric for plots", numeric_cols, index=0 if numeric_cols else None)

    if col_x and col_y:
        st.subheader(f"Bar chart of mean {col_y} by {col_x}")
        grouped = df_work.groupby(col_x)[col_y].mean().reset_index().sort_values(col_y, ascending=False)
        fig = px.bar(grouped, x=col_x, y=col_y, labels={col_x:col_x, col_y:col_y}, title=f"Mean {col_y} by {col_x}")
        st.plotly_chart(fig, use_container_width=True)

    if col_survived and col_x:
        st.subheader(f"Survival rate by {col_x}")
        surv = df_work.groupby(col_x)[col_survived].mean().reset_index().sort_values(col_survived, ascending=False)
        surv[col_survived] = (surv[col_survived]*100).round(2)
        fig2 = px.bar(surv, x=col_x, y=col_survived, labels={col_survived:"Survival %"}, title=f"Survival % by {col_x}")
        st.plotly_chart(fig2, use_container_width=True)

    if col_y:
        st.subheader(f"Distribution of {col_y}")
        fig3 = px.histogram(df_work, x=col_y, nbins=30, marginal="box", title=f"Distribution of {col_y}")
        st.plotly_chart(fig3, use_container_width=True)

    # Age vs Fare scatter
    if 'age' in [c.lower() for c in df_work.columns] and 'fare' in [c.lower() for c in df_work.columns]:
        age_col = next(c for c in df_work.columns if c.lower()=="age")
        fare_col = next(c for c in df_work.columns if c.lower()=="fare")
        st.subheader("Age vs Fare (scatter)")
        color_col = col_survived if col_survived else None
        fig4 = px.scatter(df_work, x=age_col, y=fare_col, color=color_col, hover_data=df_work.columns, title="Age vs Fare")
        st.plotly_chart(fig4, use_container_width=True)

    # Scatter matrix
    if len(numeric_cols) >= 3:
        st.subheader("Scatter matrix (first numeric columns)")
        sample = df_work[numeric_cols].dropna().sample(n=min(300, len(df_work)))
        fig5 = px.scatter_matrix(sample, dimensions=sample.columns[:4].tolist(), title="Scatter matrix (first 4 numeric columns)")
        st.plotly_chart(fig5, use_container_width=True)

# ---------- Explore & Download ----------
with tabs[4]:
    st.header("Explore & Download")
    st.markdown("Use the quick tools below to create common features and download the result. These actions operate on the filtered data.")
    fe_df = df_work.copy()
    if col_name:
        if st.button("Extract Title from Name"):
            fe_df["title"] = fe_df[col_name].apply(title_from_name)
            st.success("Title column created.")
            st.dataframe(fe_df[["title"]].value_counts().rename_axis("title").reset_index(name="count"))

    if st.button("Create family_size (SibSp + Parch + 1)"):
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

    st.markdown("**Download** the engineered dataset below:")
    csv = fe_df.to_csv(index=False)
    st.download_button("Download engineered dataset (CSV)", data=csv, file_name="titanic_engineered.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Filtered data (preview)")
    st.dataframe(df_work.head(30))
    st.download_button("Download filtered dataset (CSV)", data=df_work.to_csv(index=False), file_name="titanic_filtered.csv", mime="text/csv")

# ---------- About ----------
with tabs[5]:
    st.header("About this app")
    st.markdown('''
    This Streamlit application was created to provide a **friendly, visual** exploration of the Titanic dataset.
    It keeps the full analysis available for analysts while also offering a simplified mode for non-technical users.
    ''')
    st.markdown("**Helpful next steps for non-technical users:**")
    st.markdown("- Try different filters (sex, class, embark) and see how the survival rate changes.\n- Use the simple charts tab to get quick visual answers.\n- If you want to share your findings, use the download buttons to export CSV files.")
    st.markdown("**Deployment tips:** Save this file as `titanic_app_pretty.py` and run with `streamlit run titanic_app_pretty.py`.")
    st.markdown("**Need more polish?** I can add a custom color palette, a logo, or a guided walkthrough text for each chart. Tell me what style you prefer.")

