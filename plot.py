"""
plot
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(y, x=None, title='title', x_label='x', y_label='y', mycolor='#D1DDC5'):
    """
    x, y: 1d arrays with the same size.
    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(mycolor)
    ax.patch.set_facecolor(mycolor)
    if x:
        ax.plot(x, y)
    else:
        ax.plot(y)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
    
def plot_modes(Modes, au, t, compare_residual_with_noise=True, x_label='time', ylabel='magnitude', mycolor='#D1DDC5'):
    assert Modes.shape[0] < Modes.shape[1]
    assert Modes.shape[1] == au.size
    N = Modes.shape[0]    

    if compare_residual_with_noise:
        fig, ax = plt.subplots(N+2, 1)
        noise = 0.1*np.random.normal(size=au.size)
        noise[noise>0.3] = 0.3
        noise[noise<-0.3] = -0.3
        noise[0] = 0.5
        noise[1] = -0.5
        ax[N+1].set_title('compare with noise')
        ax[N+1].set_facecolor(mycolor)
        ax[N+1].plot(t, noise, color='gray')       
    else:
        fig, ax = plt.subplots(N+1, 1)
  
    ax[0].set_title('original signal')
    ax[0].set_facecolor(mycolor)
    ax[0].plot(t, au, color='green')
    
    for i in range(1, N):
        ax[i].set_title(f'mode {i}')     
        ax[i].set_facecolor(mycolor)
        ax[i].plot(t, Modes[i-1, :]) 

    ax[N].set_title('residual')
    ax[N].set_facecolor(mycolor)
    ax[N].plot(t, Modes[N-1, :])
    
    fig.set_facecolor(mycolor)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.show()

def plot_modes_residual(Modes, res, au, t, compare_residual_with_noise=True, x_label='time', ylabel='magnitude', mycolor='#D1DDC5'):
    assert Modes.shape[0] < Modes.shape[1]
    assert Modes.shape[1] == res.size == au.size
    N = Modes.shape[0]   

    if compare_residual_with_noise:
        fig, ax = plt.subplots(N+3, 1)
        noise = 0.1*np.random.normal(size=au.size)
        noise[noise>0.3] = 0.3
        noise[noise<-0.3] = -0.3
        noise[0] = 0.5
        noise[1] = -0.5
        ax[N+2].set_title('compare with noise')
        ax[N+2].set_facecolor(mycolor)
        ax[N+2].plot(t, noise, color='gray')       
    else:
        fig, ax = plt.subplots(N+1, 1)
  
    ax[0].set_title('original signal')
    ax[0].set_facecolor(mycolor)
    ax[0].plot(t, au, color='green')
    
    for i in range(1, N+1):
        ax[i].set_title(f'mode {i}')     
        ax[i].set_facecolor(mycolor)
        ax[i].plot(t, Modes[i-1, :]) 

    ax[N+1].set_title('residual')
    ax[N+1].set_facecolor(mycolor)
    ax[N+1].plot(t, res)
    
    fig.set_facecolor(mycolor)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.show()  

def plot_pdf(x, title='probability density function', x_label='x', y_label='density', mycolor='#D1DDC5'):
    import seaborn as sns
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
