import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

def scatter_plot_scores(csv_path):
    df = pd.read_csv(csv_path)
    fig = px.scatter(x=df['amount_score'], y=df['color_score'])
    fig.show(renderer='browser')

def plot_level1_results(dir_path):
    fig = go.Figure()
    fig.update_layout(
        title='Plan iteration vs score',
        xaxis_title='Plan iteration',
        yaxis_title='Score'
    )
    fig.update_xaxes(
        dtick=1,
        tick0=0
    )
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        df = pd.read_csv(file_path)
        color_scores = df['color_score']
        amount_scores = df['amount_score']
        weighted_scores = 0.9 * color_scores + 0.1 * amount_scores
        line_trace = go.Scatter(
            x=df['iteration'],
            y=weighted_scores, 
            mode='lines+markers',
            name=os.path.splitext(file)[0]
        )
        fig.add_trace(line_trace)

    fig.show()
        


if __name__ == '__main__':
    # scatter_plot_scores('results/l1-no-refine.csv')
    plot_level1_results('results/level1')