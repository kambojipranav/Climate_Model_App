import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM, Dense

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
st.set_page_config(page_title="NASA NLP + Forecasting", layout="wide")
st.title("🌍 PRANAV's Climate NLP App with Forecasting")

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
    st.header("🔧 Filters")
    sentiments = st.multiselect("Select Sentiments", df['transformer_sentiment'].unique(), default=list(df['transformer_sentiment'].unique()))
    n_clusters = st.slider("Number of Topics", 2, 10, 5)

# ---------- Filter + Topic Modeling ----------
filtered = df[df['transformer_sentiment'].isin(sentiments)].copy()
clusters, top_words = run_topic_model(filtered['clean_text'], n_clusters)
filtered['topic'] = clusters

# ---------- Display Top Keywords ----------
st.subheader("🧠 Topic Keywords")
for t in range(n_clusters):
    st.markdown(f"**Topic {t}**: {', '.join(top_words[t])}")

# ---------- Monthly Comment Volume ----------
st.subheader("📊 Monthly Comment Volume")
monthly_volume = df.groupby('month').size().reset_index(name='count')
fig = px.bar(monthly_volume, x='month', y='count', title="Comments per Month", color_discrete_sequence=["#00ffd5"])
st.plotly_chart(fig, use_container_width=True)

# ---------- Topic Trend Split by Sentiment ----------
st.subheader("📈 Topic Trends Over Time")
for s in sentiments:
    st.markdown(f"### {s.capitalize()} Sentiment")
    sub_df = filtered[filtered['transformer_sentiment'] == s]
    trend = sub_df.groupby(['month', 'topic']).size().reset_index(name='count')
    pivot = trend.pivot(index='month', columns='topic', values='count').fillna(0)
    percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig = px.line(percent.reset_index(), x='month', y=percent.columns, markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------- LSTM Forecasting ----------
st.subheader("🔮 LSTM Forecasting for Next Month's Topics")
# Monthly count per topic
pivot = filtered.groupby(['month', 'topic']).size().unstack(fill_value=0)
data = pivot.values.astype(np.float32)

X, y = data[:-1], data[1:]
X = X.reshape((X.shape[0], 1, X.shape[1]))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

pred = model.predict(X[-1].reshape(1, 1, -1))[0]
pred_series = pd.Series(pred, index=pivot.columns)
st.dataframe(pred_series.round(1).sort_values(ascending=False).rename("Predicted Count"))

# ---------- Sample Viewer ----------
st.subheader("📝 Sample Comments")
selected_topic = st.selectbox("Choose a Topic", sorted(filtered['topic'].unique()))
st.dataframe(filtered[filtered['topic'] == selected_topic][['date', 'clean_text', 'transformer_sentiment']].head(10))

# ---------- Download ----------
st.download_button("📥 Download All Output", filtered.to_csv(index=False), file_name="topic_output.csv")
st.markdown('<div class="footer">2025 © Pranav The King</div>', unsafe_allow_html=True)

