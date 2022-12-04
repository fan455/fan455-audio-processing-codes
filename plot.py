import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pdf(x, title='probability density function', x_label='x', y_label='density', mycolor='#D1DDC5'):
    sns.set(rc={'axes.facecolor':mycolor, 'axes.edgecolor':'grey', 'figure.facecolor':mycolor})
    ax = sns.kdeplot(data=x)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_xy(x, y, title='title', x_label='x', y_label='y', mycolor='#D1DDC5'):
    """
    x, y: 1d ndarrays with the same sizes.
    fast plot of x and y
    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(mycolor)
    ax.patch.set_facecolor(mycolor)
    ax.plot(x, y)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_au_mono(au, sr, title='title', mycolor='#D1DDC5'):
    t = np.arange(0, au.size)/sr
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(mycolor)
    ax.patch.set_facecolor(mycolor)
    ax.plot(t, au)
    ax.set_title(title)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_y(y, title='title', x_label='x', y_label='y', mycolor='#D1DDC5'):
    """
    x is the index of y.
    x, y: 1d ndarrays with the same sizes.
    fast plot of x and y
    """
    x = np.arange(0, y.size)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(mycolor)
    ax.patch.set_facecolor(mycolor)
    ax.plot(x, y)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
