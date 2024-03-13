import numpy as np
import matplotlib.pyplot as plt

def view(M, d, which, title=None, x_tick_labels=None, y_tick_labels=None,
         x_label=None):
    
    m,n = np.shape(M)
    
    fig, ax = plt.subplots()
    im = ax.imshow(M, aspect="auto", origin="lower")
    
    x_ticks = np.linspace(0, n, 5)
    if x_tick_labels == None:
        x_tick_labels = np.linspace(0, 1, 5)
    y_ticks = np.linspace(0, m, 5)
    if y_tick_labels == None:
        y_tick_labels = np.round(np.linspace(np.min(d), np.max(d), 5), 2)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    if which == 0:
        ax.set_title(title)
        ax.set_xlabel("Scales")
        ax.set_ylabel("Scales")
        
    elif which == 1:
        if title == None:
            ax.set_title("Spectrum")
        else:
            ax.set_title(title)
        if x_label == None:
            ax.set_xlabel(r"$z$")
        else:
            ax.set_xlabel(x_label)
        ax.set_ylabel("Scales")
    else:
        if title == None:
            ax.set_title("Local ACF")
        else:
            ax.set_title(title)
        if x_label == None:
            ax.set_xlabel(r"$z$")
        else:
            ax.set_xlabel(x_label)
        ax.set_ylabel("Lag")
    
    plt.colorbar(im, ax=ax)
    
    return None