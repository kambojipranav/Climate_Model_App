import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
import plotly.express as px
import os

# App Title
st.set_page_config(page_title="Climate Change Sentiment & Topics", layout="wide")
st.title("🌍 Climate Change Sentiment & Topic Modeling (NASA Facebook Comments)")

# Load Data and Model
@st.cache_resource
def load_data():
    df = pd.read_csv("model/sentiment.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period("M").astype(str)
    return df

@st.cache_resource
def load_topic_model():
    return BERTopic.load("model/topic_model.bin")

df = load_data()
topic_model = load_topic_model()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    sentiments = st.multiselect("Select Sentiment", df['transformer_sentiment'].unique(), default=df['transformer_sentiment'].unique())
    top_n = st.slider("Top N Topics to Display", 3, 10, 5)

filtered_df = df[df['transformer_sentiment'].isin(sentiments)]

# Topic trend analysis
st.subheader("📈 Topic Trends Over Time")
topic_trend = filtered_df.groupby(['month', 'topic']).size().reset_index(name='count')
pivot = topic_trend.pivot(index='month', columns='topic', values='count').fillna(0)
pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
top_topics = pivot.sum().sort_values(ascending=False).head(top_n).index

fig = px.line(pivot_percent[top_topics].reset_index(), x='month', y=top_topics, markers=True)
st.plotly_chart(fig, use_container_width=True)

# Topic keywords section
st.subheader("🧠 Topic Keywords")
for topic_id in top_topics:
    words = topic_model.get_topic(topic_id)
    st.markdown(f"**Topic {topic_id}**: " + ", ".join([w[0] for w in words]))

# Individual comment viewer
st.subheader("📝 Sample Comments by Topic")
selected_topic = st.selectbox("Choose a Topic", top_topics)
st.dataframe(filtered_df[filtered_df['topic'] == selected_topic][['date', 'clean_text', 'transformer_sentiment']].head(10))

# Optional: Download CSV
st.download_button("📥 Download Processed Data", df.to_csv(index=False), file_name="nasa_topic_output.csv")

# Email section (optional)
with st.expander("📧 Email this analysis"):
    to_email = st.text_input("Enter recipient email")
    if st.button("Send Email"):
        from utils.email_helper import send_email
        send_email(to_email, subject="NASA Climate Analysis", attachment_path="model/sentiment.csv")
        st.success("Email sent!")
