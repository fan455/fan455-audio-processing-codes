"""
fast plot
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fft_p(p, title='fft frequency phases', x_label='freq in samples', y_label='phase', mycolor='#D1DDC5'):
    f = np.arange(p.shape[0])
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(mycolor)
    ax.patch.set_facecolor(mycolor)
    ax.plot(f, p)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_fft_m(m, title='fft frequency magnitudes', x_label='freq in samples', y_label='magnitude', mycolor='#D1DDC5'):
    f = np.arange(m.shape[0])
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(mycolor)
    ax.patch.set_facecolor(mycolor)
    ax.plot(f, m)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_pdf(x, title='probability density function', x_label='x', y_label='density', mycolor='#D1DDC5'):
    """
    Plot the probability density function of x.
    x: 1d array.
    """
    sns.set(rc={'axes.facecolor':mycolor, 'axes.edgecolor':'grey', 'figure.facecolor':mycolor})
    ax = sns.kdeplot(data=x)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_xy(x, y, title='title', x_label='x', y_label='y', mycolor='#D1DDC5'):
    """
    x, y: 1d arrays with the same size.
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
    """
    Plot mono audio array.
    au: 1d audio array.
    sr: sample rate.
    """
    t = np.arange(au.size)/sr
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
    Plot array y, x representing the indices of y.
    y: 1d array.
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
