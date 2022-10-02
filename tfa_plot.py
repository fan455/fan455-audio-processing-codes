"""
Time-Frequency Analysis plot.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_stft_m(f, t, m, win_idx=0):
    time = t[win_idx]
    x, y = f, m[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('frequency magnitude')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_p(f, t, p, win_idx=0):
    time = t[win_idx]
    x, y = f, p[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('frequency phase')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_z(f, t, z, win_idx=0):
    time = t[win_idx]
    z = z[:, win_idx]
    x, y1, y2 = f, z.real, z.imag
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    fig.patch.set_facecolor('#D1DDC5')
    ax1.patch.set_facecolor('#D1DDC5')
    ax2.patch.set_facecolor('#D1DDC5')
    plt.xlabel('frquency')
    ax1.set_title(f'time = {time}s' + '\n' + 'real part')
    ax2.set_title('imaginary part')
    ax1.grid(color='grey', linewidth='1', linestyle='-.')
    ax2.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_r(f, t, r, win_idx=0):
    time = t[win_idx]
    x, y = f, r[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('real part')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_i(f, t, i, win_idx=0):
    time = t[win_idx]
    x, y = f, i[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('imaginary part')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_psd(f, Pxx):
    f = f.astype(np.int16)
    x, y = f, Pxx
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'Power Spectral Dessity')
    plt.xlabel('frquency')
    plt.ylabel('psd')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
