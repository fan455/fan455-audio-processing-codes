"""
Plot
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot(y, x=None, title='title', xlabel='x', ylabel='y', bgcolor='#D1DDC5', **kwargs):
    """
    x, y: 1d arrays with the same size.
    """
    fig, ax = plt.subplots()
    fig.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)
    if x is None:
        ax.plot(y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='top')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def subplots(y, x, nrows=None, ncols=1, yaxis=1, title=None, subtitle=None, xlabel='x', ylabel='y', bgcolor='#D1DDC5', **kwargs):
    assert x.ndim == 1 and y.ndim == 2
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    if nrows == None:
        nrows = y.shape[0]
    assert nrows*ncols == y.shape[0]
    N = nrows*ncols
    fig, ax = plt.subplots(nrows, ncols)
    if subtitle:
        if type(subtitle) == str:
            for i in range(N):
                ax[i].set_title(f'{subtitle} {i+1}', fontsize='medium')     
                ax[i].set_facecolor(bgcolor)
                ax[i].plot(x, y[i, :], **kwargs)             
        elif type(subtitle) == list or type(subtitle) == tuple:
            assert len(subtitle) == N
            for i in range(N):
                ax[i].set_title(subtitle[i], fontsize='medium')     
                ax[i].set_facecolor(bgcolor)
                ax[i].plot(x, y[i, :], **kwargs)
        else:
            raise ValueError('Your subtitle type {type(subtitle)} is not supported.')
    else:
        ax[i].set_facecolor(bgcolor)
        ax[i].plot(x, y[i, :], **kwargs)

    fig.set_facecolor(bgcolor)
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.show()    
                
def plot_modes(Modes, au, t, compare_with_noise=True, title=None, xlabel='time', ylabel='magnitude', bgcolor='#D1DDC5', **kwargs):
    assert Modes.shape[0] < Modes.shape[1]
    assert Modes.shape[1] == au.size
    N = Modes.shape[0]    

    if compare_with_noise:
        fig, ax = plt.subplots(N+2, 1)
        noise = 0.1*np.random.normal(size=au.size)
        noise[noise>0.3] = 0.3
        noise[noise<-0.3] = -0.3
        noise[0] = -0.5
        noise[-1] = 0.5
        ax[N+1].set_title('compare with Gaussian noise', fontsize='medium')
        ax[N+1].set_facecolor(bgcolor)
        ax[N+1].plot(t, noise, color='gray', **kwargs)       
    else:
        fig, ax = plt.subplots(N+1, 1)
  
    ax[0].set_title('original signal', fontsize='medium')
    ax[0].set_facecolor(bgcolor)
    ax[0].plot(t, au, color='green', **kwargs)
    
    for i in range(1, N):
        ax[i].set_title(f'mode {i}', fontsize='medium')     
        ax[i].set_facecolor(bgcolor)
        ax[i].plot(t, Modes[i-1, :], **kwargs) 

    ax[N].set_title('residual', fontsize='medium')
    ax[N].set_facecolor(bgcolor)
    ax[N].plot(t, Modes[N-1, :], **kwargs)
    
    fig.set_facecolor(bgcolor)
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.show()

def plot_modes_residual(Modes, res, au, t, compare_with_noise=True, title=None, xlabel='time', ylabel='magnitude', bgcolor='#D1DDC5', **kwargs):
    assert Modes.shape[0] < Modes.shape[1]
    assert Modes.shape[1] == res.size == au.size
    N = Modes.shape[0]   

    if compare_with_noise:
        fig, ax = plt.subplots(N+3, 1)
        noise = 0.1*np.random.normal(size=au.size)
        noise[noise>0.3] = 0.3
        noise[noise<-0.3] = -0.3
        noise[0] = -0.5
        noise[-1] = 0.5
        ax[N+2].set_title('compare with Gaussian noise', fontsize='medium')
        ax[N+2].set_facecolor(bgcolor)
        ax[N+2].plot(t, noise, color='gray', **kwargs)       
    else:
        fig, ax = plt.subplots(N+1, 1)
  
    ax[0].set_title('original signal', fontsize='medium')
    ax[0].set_facecolor(bgcolor)
    ax[0].plot(t, au, color='green', **kwargs)
    
    for i in range(1, N+1):
        ax[i].set_title(f'mode {i}', fontsize='medium')     
        ax[i].set_facecolor(bgcolor)
        ax[i].plot(t, Modes[i-1, :], **kwargs) 

    ax[N+1].set_title('residual', fontsize='medium')
    ax[N+1].set_facecolor(bgcolor)
    ax[N+1].plot(t, res, **kwargs)
    
    fig.set_facecolor(bgcolor)
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.show()  

def plot_pdf(x, title='probability density function', xlabel='x', ylabel='density', bgcolor='#D1DDC5'):
    """
    Plot the probability density function of x.
    x: 1d array.
    """
    sns.set(rc={'axes.facecolor':bgcolor, 'axes.edgecolor':'grey', 'figure.facecolor':bgcolor})
    ax = sns.kdeplot(data=x)
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='top')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_au_mono(au, sr, title='title', bgcolor='#D1DDC5'):
    """
    Plot mono audio array.
    au: 1d audio array.
    sr: sample rate.
    """
    t = np.arange(au.size)/sr
    fig, ax = plt.subplots()
    fig.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)
    ax.plot(t, au)
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
