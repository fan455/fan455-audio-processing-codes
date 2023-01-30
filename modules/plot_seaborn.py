"""
Plot
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pdf(x, title='probability density function', xlabel='x', ylabel='density', bgcolor='#D1DDC5', grid=True):
    """
    Plot the probability density function of x.
    x: 1d array.
    """
    sns.set(rc={'axes.facecolor':bgcolor, 'axes.edgecolor':'grey', 'figure.facecolor':bgcolor})
    ax = sns.kdeplot(data=x)
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

