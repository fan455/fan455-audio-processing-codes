import numpy as np
import matplotlib.pyplot as plt

def plot_xy(x, y, title='title', x_label='x', y_label='y'):
    """
    x, y: 1d ndarrays with the same sizes.
    fast plot of x and y
    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_y(y, title='title', x_label='x', y_label='y'):
    """
    x is the index of y.
    x, y: 1d ndarrays with the same sizes.
    fast plot of x and y
    """
    x = np.arange(0, y.size)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
