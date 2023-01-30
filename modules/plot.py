"""
Plot
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(y, x=None, title='title', xlabel='x', ylabel='y', grid=True, bgcolor='#D1DDC5', **kwargs):
    """
    x, y: 1d arrays with the same size.
    """
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    if x is None:
        ax.plot(y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_xint(y, x=None, title='title', xlabel='x', ylabel='y', grid=True, annotate_x=False, bgcolor='#D1DDC5'):
    """
    x, y: 1d arrays with the same size.
    """
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    if x is None:
        x = np.arange(y.size)
    else:
        assert x.dtype == int
    ax.plot(x, y, color='tab:grey', linestyle='--', marker='.', markersize=12.0, mec='black', mfc='black')
    if annotate_x:
        for i in range(x.size):
            plt.annotate(f'{x[i]}', (x[i], y[i]), fontsize='small')
    else:
        pass
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
    
def subplots(y, x, nrows=None, ncols=1, yaxis=1, title=None, subtitle=None, xlabel='x', ylabel='y', grid=False, bgcolor='#D1DDC5', **kwargs):
    assert x.ndim == 1 and y.ndim == 2
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    if nrows == None:
        nrows = y.shape[0]
    assert nrows*ncols == y.shape[0]
    N = nrows*ncols
    fig, ax = plt.subplots(nrows, ncols, facecolor=bgcolor)
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

    if grid:
        for i in range(N):
            ax[i].grid(color='grey', linewidth='1', linestyle='-.')
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()    
                
def plot_modes(Modes, au, t, compare_with_noise=True, title=None, xlabel='time', ylabel='magnitude', grid=False, bgcolor='#D1DDC5', **kwargs):
    assert Modes.shape[0] < Modes.shape[1]
    assert Modes.shape[1] == au.size
    N = Modes.shape[0]    

    if compare_with_noise:
        fig, ax = plt.subplots(N+2, 1, facecolor=bgcolor)
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
    
    if grid:
        for i in range(N+2):
            ax[i].grid(color='grey', linewidth='1', linestyle='-.')
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_modes_residual(Modes, res, au, t, compare_with_noise=True, title=None, xlabel='time', ylabel='magnitude', grid=False, bgcolor='#D1DDC5', **kwargs):
    assert Modes.shape[0] < Modes.shape[1]
    assert Modes.shape[1] == res.size == au.size
    N = Modes.shape[0]   

    if compare_with_noise:
        fig, ax = plt.subplots(N+3, 1, facecolor=bgcolor)
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
    
    if grid:
        for i in range(N+3):
            ax[i].grid(color='grey', linewidth='1', linestyle='-.')
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()  

def plot_au_mono(au, sr, title='title', grid=True, bgcolor='#D1DDC5'):
    """
    Plot mono audio array.
    au: 1d audio array.
    sr: sample rate.
    """
    t = np.arange(au.size)/sr
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    ax.plot(t, au)
    plt.title(title)
    plt.xlabel('time', loc='right')
    plt.ylabel('amplitude', loc='center')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
