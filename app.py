import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict

# ---------- Custom Styling ----------
st.markdown("""
<style>
.footer {
    font-size: 32px;           /* Make it big */
    font-weight: bold;
    color: #ffffff;            /* You can change this */
    text-align: center;        /* Center the text */
    padding: 20px;
    margin-top: 40px;
}

body {
    background: linear-gradient(to right, #00FFFF, #FFC300);
    color: white;
    font-size: 20px; /* increase global font size */
}
.stApp {
    background: linear-gradient(to right, #eb1d02, #6790f0);
}
h1 {
    font-size: 40px !important;
    color: #00ffd5;
}
h2 {
    font-size: 32px !important;
}
h3 {
    font-size: 26px !important;
}
h4 {
    font-size: 22px !important;
}
div[data-testid="stMarkdownContainer"] p {
    font-size: 18px !important;
}
.css-1d391kg {
    background-color: rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- Page Setup ----------
st.set_page_config(page_title="NASA NLP App", layout="wide")
st.title("🌍 PRANAV's Climate NLP App")

# ---------- Load and Preprocess ----------
@st.cache_data
def load_data():
    df = pd.read_csv("sentiment.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period("M").astype(str)
    return df

@st.cache_resource
def run_topic_model(texts, n_clusters=5):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(texts)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)
    top_words = {
        i: [tfidf.get_feature_names_out()[j] for j in model.cluster_centers_[i].argsort()[-10:][::-1]]
        for i in range(n_clusters)
    }
    return clusters, top_words

# ---------- Load Data ----------
df = load_data()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧪 Filters")
    sentiments = st.multiselect("Choose Sentiments", df['transformer_sentiment'].unique(), default=list(df['transformer_sentiment'].unique()))
    n_clusters = st.slider("Number of Topics", 2, 10, 5)

# ---------- Filter + Topic Modeling ----------
filtered = df[df['transformer_sentiment'].isin(sentiments)].copy()
clusters, top_words = run_topic_model(filtered['clean_text'], n_clusters)
filtered['topic'] = clusters

# ---------- Topic Keywords ----------
st.subheader("🧠 Top Topic Keywords")
for i in range(n_clusters):
    st.markdown(f"**Topic {i}**: {', '.join(top_words[i])}")

# ---------- Monthly Comment Volume ----------
st.subheader("📊 Monthly Comment Volume")
monthly_volume = df.groupby('month').size().reset_index(name='count')
fig = px.bar(monthly_volume, x='month', y='count', title="Comments per Month", color_discrete_sequence=["#FF6F61"])
st.plotly_chart(fig, use_container_width=True)

# ---------- Topic Trends by Sentiment ----------
st.subheader("📈 Topic Trends Over Time (by Sentiment)")
for s in sentiments:
    st.markdown(f"### {s.capitalize()} Sentiment")
    sub_df = filtered[filtered['transformer_sentiment'] == s]
    trend = sub_df.groupby(['month', 'topic']).size().reset_index(name='count')
    pivot = trend.pivot(index='month', columns='topic', values='count').fillna(0)
    percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig = px.line(percent.reset_index(), x='month', y=percent.columns, markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Sample Comments Viewer ----------
st.subheader("📝 View Sample Comments by Topic")
selected_topic = st.selectbox("Select a Topic", sorted(filtered['topic'].unique()))
st.dataframe(filtered[filtered['topic'] == selected_topic][['date', 'clean_text', 'transformer_sentiment']].head(10))

# ---------- Download Output ----------
st.download_button("📥 Download CSV", filtered.to_csv(index=False), file_name="topic_output.csv")

# ---------- Footer ----------
st.markdown('<div class="footer">2025 © Pranav The King 👑</div>', unsafe_allow_html=True)
