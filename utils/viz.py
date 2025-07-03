import pandas as pd
import plotly.express as px

def plot_topic_trend(df, top_n=5):
    df['month'] = df['date'].dt.to_period("M").astype(str)
    grouped = df.groupby(['month', 'topic']).size().reset_index(name='count')
    pivot = grouped.pivot(index='month', columns='topic', values='count').fillna(0)
    percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    top_topics = pivot.sum().sort_values(ascending=False).head(top_n).index
    fig = px.line(percent[top_topics].reset_index(), x='month', y=top_topics, markers=True)
    return fig
