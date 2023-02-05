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

def plot_xint(y, x=None, title='title', xlabel='x', ylabel='y', \
              grid=True, annotate_x=False, bgcolor='#D1DDC5'):
    """
    x, y: 1d arrays with the same size.
    """
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    if x is None:
        x = np.arange(y.size)
    else:
        assert x.dtype == int
    ax.plot(x, y, color='tab:grey', linestyle='--', \
            marker='.', markersize=12.0, mec='black', mfc='black')
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
    
def subplots(y, x, nrows=None, ncols=1, yaxis=1, title=None, subtitle=None, \
             xlabel='x', ylabel='y', grid=False, bgcolor='#D1DDC5', **kwargs):
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

def plot_scale(y, x, xscale='log', yscale=None, xscale_kwargs=None, yscale_kwargs=None, \
               title='title', xlabel='x', ylabel='y', grid=True, bgcolor='#D1DDC5', **kwargs):
    # Plot with x and/or y axis scaled. By default x is scaled by "log10".
    # The types of xscale_kwargs and yscale_kwargs should be "dict", so use the "dict()" function.
    # Please refer to "matplotlib.scale".
    # xscale or yscale: 'asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog'
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    if xscale is not None:
        if xscale_kwargs is None:
            ax.set_xscale(xscale)
        else:
            ax.set_xscale(xscale, **xscale_kwargs)
    if yscale is not None:
        if yscale_kwargs is None:
            ax.set_yscale(yscale)
        else:
            ax.set_yscale(yscale, **yscale_kwargs)
    ax.plot(x, y, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_frm(frm, f, base=10, linthresh=10, linscale=10, subs=None, \
             delete_zero=True, grid=True, bgcolor='#D1DDC5', **kwargs):
    # Plot frequency response: magnitude.
    # base, linthresh linscale and subs are keyword arguments for 'matplotlib.scale.SymmetricalLogScale'.
    # In case of zero division warning: (old_settings =) np.seterr(divide='ignore')
    if delete_zero:
        frm, f = frm[1:], f[1:]
    frm = 20*np.log10(frm)
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    ax.set_xscale('symlog', base=base, linthresh=linthresh, subs=subs, linscale=linscale)
    ax.plot(f, frm, **kwargs)
    plt.title('frequency response: magnitude')
    plt.xlabel(xlabel, loc='frequency (Hz)')
    plt.ylabel(ylabel, loc='magnitude (dB)')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_frp(frp, f, base=10, linthresh=10, linscale=10, subs=None, \
             delete_zero=False, grid=True, bgcolor='#D1DDC5', **kwargs):
    # Plot frequency response: phase.
    # base, linthresh linscale and subs are keyword arguments for 'matplotlib.scale.SymmetricalLogScale'.
    # In case of zero division warning: (old_settings =) np.seterr(divide='ignore')
    if delete_zero:
        frp, f = frp[1:], f[1:]
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    ax.set_xscale('symlog', base=base, linthresh=linthresh, subs=subs, linscale=linscale)
    ax.plot(f, frp, **kwargs)
    plt.title('frequency response: phase')
    plt.xlabel(xlabel, loc='frequency (Hz)')
    plt.ylabel(ylabel, loc='phase (rad)')
    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
                
def plot_modes(Modes, t, au=None, res=None, compare_with_noise=True, title=None, \
               xlabel='time', ylabel='magnitude', grid=False, bgcolor='#D1DDC5', **kwargs):
    assert Modes.shape[1] == t.size
    assert Modes.shape[0] < Modes.shape[1]
    N = Modes.shape[0]
    nrows = N
    if au is not None:
        assert au.size == t.size
        nrows += 1
    if res is not None:
        assert res.size == t.size
        nrows += 1
    if compare_with_noise:
        nrows += 1

    fig, ax = plt.subplots(nrows=nrows, ncols=1, facecolor=bgcolor)
    
    nadj = 0
    if au is not None:
        ax[0].set_title('original signal', fontsize='medium')
        ax[0].set_facecolor(bgcolor)
        ax[0].plot(t, au, color='green', **kwargs)
        nadj += 1
        
    for i in range(0, N):
        ax[i+nadj].set_title(f'mode {i+1}', fontsize='medium')     
        ax[i+nadj].set_facecolor(bgcolor)
        ax[i+nadj].plot(t, Modes[i, :], **kwargs)

    if res is not None:  
        ax[N+nadj].set_title('residual', fontsize='medium')
        ax[N+nadj].set_facecolor(bgcolor)
        ax[N+nadj].plot(t, res, **kwargs)
        nadj += 1
    else:
        ax[N-1+nadj].set_title('residual', fontsize='medium')
    
    if compare_with_noise:
        noise = 0.1*np.random.normal(size=t.size)
        noise[noise>0.3] = 0.3
        noise[noise<-0.3] = -0.3
        noise[0] = -0.5
        noise[-1] = 0.5
        ax[N+nadj].set_title('compare with Gaussian noise', fontsize='medium')
        ax[N+nadj].set_facecolor(bgcolor)
        ax[N+nadj].plot(t, noise, color='gray', **kwargs) 
    
    if grid:
        for i in range(nrows):
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
