import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def violin_plot(fig, df, direction, row, col, pointpos_male, pointpos_female, show_legend):


    for i in range(0,len(pd.unique(df['condition']))):
        fig.add_trace(go.Violin(x=df['condition'][(df['sex'] == 'M') &
                                                  (df['condition'] == pd.unique(df['condition'])[i])],
                                y=df[direction][(df['sex'] == 'M')&
                                              (df['condition'] == pd.unique(df['condition'])[i])],
                                legendgroup='M', name='M',
                                side='negative',
                                line_color='lightseagreen',
                                showlegend=show_legend,), row=row, col=col
                      )

        fig.add_trace(go.Violin(x=df['condition'][(df['sex'] == 'F') &
                                                  (df['condition'] == pd.unique(df['condition'])[i])],
                                y=df[direction][(df['sex'] == 'F')&
                                              (df['condition'] == pd.unique(df['condition'])[i])],
                                legendgroup='F', name='F',
                                side='positive',
                                line_color='mediumpurple',
                                showlegend=show_legend, ), row=row, col=col
                      )

    return fig