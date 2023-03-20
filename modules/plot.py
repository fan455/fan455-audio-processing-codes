# fast plot by wrapping matplotlib

import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['figure.dpi'] = 150
#plt.rcParams['savefig.dpi'] = 150
#plt.rcParams['font.family'] ='sans-serif'
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams.update({'font.size': 11.5})

def plot(y, x=None, title='title', xlabel='x', ylabel='y', xtick=None, ytick=None, \
         xticklabel=None, yticklabel=None, xtick_kwargs=None, ytick_kwargs=None, \
         grid=True, bgcolor='#D1DDC5', **kwargs):
    # x and y are both 1d arrays with the same size.
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    if x is None:
        ax.plot(y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)

    if xtick is not None:
        ax.set_xticks(xtick)
        if xticklabel is not None:
            ax.set_xticklabels(xticklabel)
        if xtick_kwargs is not None:
            ax.tick_params(axis='x', **xtick_kwargs)

    if ytick is not None:
        ax.set_yticks(ytick)
        if yticklabel is not None:
            ax.set_yticklabels(yticklabel)
        if ytick_kwargs is not None:
            ax.tick_params(axis='y', **ytick_kwargs)

    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
    plt.show()

def plot_multi(y, x, yaxis=1, linelabels=None, kwargslist=None, legendkwargs=None, \
               title='title', xlabel='x', ylabel='y', xtick=None, ytick=None, \
               xticklabel=None, yticklabel=None, xtick_kwargs=None, ytick_kwargs=None, \
               grid=True, bgcolor='#D1DDC5'):
    # Plot multiple lines in the same graph.
    # x is a 1d array with shape (npoints,).
    # y is a 2d array with shape (N, npoints) if yaxis=1, or shape (npoints, N) if yaxis=0.
    # kwargslist is a list containing N dictionaries with kwargs sent to ax.plot() corresponding to the N lines.
    # linelabels is a list containing N strings setting the labels of lines.
    # legendkwargs is a dictionary of legend parameters, sent to ax.legend()
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    N = y.shape[0]
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    
    if linelabels is None:
        linelabels = list(f'y{i}' for i in range(N))
        
    if kwargslist:
        for i in range(N):
            ax.plot(x, y[i, :], **kwargslist[i])
    else:
        for i in range(N):
            ax.plot(x, y[i, :])
            
    if legendkwargs:
        ax.legend(linelabels, **legendkwargs)
    else:
        ax.legend(linelabels)

    if xtick is not None:
        ax.set_xticks(xtick)
        if xticklabel is not None:
            ax.set_xticklabels(xticklabel)
        if xtick_kwargs is not None:
            ax.tick_params(axis='x', **xtick_kwargs)

    if ytick is not None:
        ax.set_yticks(ytick)
        if yticklabel is not None:
            ax.set_yticklabels(yticklabel)
        if ytick_kwargs is not None:
            ax.tick_params(axis='y', **ytick_kwargs)

    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
    plt.show()
    
def subplots(y, x, nrows=None, ncols=1, yaxis=1, title=None, subtitle=None, \
             xlabel='x', ylabel='y', grid=False, bgcolor='#D1DDC5', **kwargs):
    # x is a 1d array, y is a 2d array with shape(n_subplots, x.size).
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    if nrows is None:
        nrows = y.shape[0]
    N = y.shape[0]
    fig, ax = plt.subplots(nrows, ncols, facecolor=bgcolor)
    ax = ax.reshape(ax.size)
    if subtitle:
        if isinstance(subtitle, str):
            for i in range(N):
                ax[i].set_title(f'{subtitle} {i+1}', fontsize='medium')     
                ax[i].set_facecolor(bgcolor)
                ax[i].plot(x, y[i, :], **kwargs)
        elif isinstance(subtitle, (list, tuple)):
            assert len(subtitle) == N
            for i in range(N):
                ax[i].set_title(subtitle[i], fontsize='medium')     
                ax[i].set_facecolor(bgcolor)
                ax[i].plot(x, y[i, :], **kwargs)
        else:
            raise ValueError('Your subtitle type {type(subtitle)} is not supported.')
    else:
        for i in range(N):
            ax[i].set_facecolor(bgcolor)
            ax[i].plot(x, y[i, :], **kwargs)

    nEmpty = nrows*ncols - N
    if nEmpty > 0:
        for i in range(nEmpty):
            ax[-nEmpty].set_facecolor(bgcolor)
            
    if grid:
        for i in range(N):
            ax[i].grid(color='grey', linewidth='0.75', linestyle='-.')
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_sc(y, x, x_trtime, yaxis=1, linelabels=['actual', 'synthetic'], \
            kwargslist=[{}, dict(linestyle='--',color='green')], \
            vlinekwargs=dict(color='grey', lw=2.0, ls='--'), \
            legendkwargs=dict(loc='lower left'), title='synthetic control', \
            xlabel='time', ylabel='y', xtick=None, ytick=None, \
            xticklabel=None, yticklabel=None, \
            xtick_kwargs=dict(labelsize=9, labelrotation=45), ytick_kwargs=None, \
            grid=True, bgcolor='#D1DDC5'):
    # Plot actual and synthetic control lines in the same graph.
    # x is a 1d array with shape (npoints,).
    # y is a 2d array with shape (N, npoints) if yaxis=1, or shape (npoints, N) if yaxis=0.
    # x_trtime (the start time of treatment), minus 1, is the x coordinate to plot a vertical line.
    # kwargslist is a list containing N dictionaries with kwargs sent to ax.plot() corresponding to the N lines.
    # linelabels is a list containing N strings setting the labels of lines.
    # legendkwargs is a dictionary of legend parameters, sent to ax.legend()    
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    N = y.shape[0]
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    
    if linelabels is None:
        linelabels = list(f'y{i}' for i in range(N))
        
    if kwargslist:
        for i in range(N):
            ax.plot(x, y[i, :], **kwargslist[i])
    else:
        for i in range(N):
            ax.plot(x, y[i, :])
            
    if legendkwargs:
        ax.legend(linelabels, **legendkwargs)
    else:
        ax.legend(linelabels)

    if xtick is not None:
        ax.set_xticks(xtick)
        if xticklabel is not None:
            ax.set_xticklabels(xticklabel)
        if xtick_kwargs is not None:
            ax.tick_params(axis='x', **xtick_kwargs)

    if ytick is not None:
        ax.set_yticks(ytick)
        if yticklabel is not None:
            ax.set_yticklabels(yticklabel)
        if ytick_kwargs is not None:
            ax.tick_params(axis='y', **ytick_kwargs)
        
    plt.axvline(x_trtime-1, **vlinekwargs)
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
    plt.show()
 
def plot_itp(y, x, points, title='title', xlabel='x', ylabel='y', \
             grid=True, bgcolor='#D1DDC5', **kwargs):
    # Plot for interpolation.
    # points is a 2d array with shape(n_points, 2), that will be annotated.
    # points = np.array([[x1, y1], [x2, y2],..., [xn, yn]])
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    ax.plot(x, y, **kwargs)
    n_points = points.shape[0]
    px, py = points[:, 0], points[:, 1]
    ax.plot(px, py, linestyle='', marker='.', markersize=9, mec='black', mfc='black')
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
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
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
    plt.show()


def plot_fscale(y, x, faxis='x', base=10, linthresh=10, linscale=0.25, subs=None, \
                title='title', xlabel='frequency (Hz)', ylabel='magnitude', \
                grid=True, bgcolor='#D1DDC5', **kwargs):
    # Plot with scaled frequency axis.
    # base, linthresh linscale and subs are keyword arguments for 'matplotlib.scale.SymmetricalLogScale'.
    # In case of zero division warning: (old_settings =) np.seterr(divide='ignore')
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    if faxis == 'x':
        ax.set_xscale('symlog', base=base, linthresh=linthresh, subs=subs, linscale=linscale)
        ax.plot(x, y, **kwargs)
        plt.xlabel('frequency (Hz)', loc='right')
        plt.ylabel(ylabel, loc='center')
    elif faxis =='y':
        ax.set_yscale('symlog', base=base, linthresh=linthresh, subs=subs, linscale=linscale)
        ax.plot(x, y, **kwargs)
        plt.xlabel(xlabel, loc='right')
        plt.ylabel('frequency (Hz)', loc='center')
    plt.title(title)
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
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
            ax[i].grid(color='grey', linewidth='0.75', linestyle='-.')
    if title:
        fig.suptitle(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_au_mono(au, sr, title='title', grid=True, bgcolor='#D1DDC5', **kwargs):
    """
    Plot mono audio array.
    au: 1d audio array of shape (nsamples,).
    sr: sample rate.
    """
    t = np.arange(au.shape[0])/sr
    plot(y, t, title=title, xlabel='time (s)', ylabel='amplitude', grid=grid, \
         bgcolor=bgcolor, **kwargs)

def plot_au_stereo(au, sr, title='title', grid=True, bgcolor='#D1DDC5', **kwargs):
    """
    Plot stereo audio array.
    au: 2d audio array of shape (nsamples, 2).
    sr: sample rate.
    """
    t = np.arange(au.shape[0])/sr
    subplots(y, t, nrows=2, ncols=1, yaxis=0, title=title, subtitle=('left channel', 'right channel'), \
             xlabel='time (s)', ylabel='amplitude', grid=True, bgcolor=bgcolor, **kwargs)
