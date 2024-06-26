import numpy as np
import matplotlib.pyplot as plt

def round_to_sig_figs(nums:np.ndarray, sig_figs:int) -> float:
    """
    Rounds a number to the specified number of significant figures.
    
    Args:
        num (float): The number to be rounded.
        sig_figs (int): The desired number of significant figures.
        
    Returns:
        float: The rounded number.
    """
    dps = sig_figs - 1 - int(np.floor(np.log10(np.max(np.abs(nums)))))
    if dps <= 0:
        return np.round(nums, dps).astype(int)
    return np.round(nums, dps)

def view(M: np.ndarray, x: np.ndarray, y: np.ndarray, regular: bool = True, num_x: int = 5, 
         num_y: int = 5, title: str = "", x_label: str = "", y_label: str = "") -> None:
    """
    Display a 2D array as an image with labeled axes and a colorbar.

    Args:
        M (np.ndarray): The 2D array to be displayed.
        x (np.ndarray): The x-axis values.
        y (np.ndarray): The y-axis values.
        regular (bool, optional): Whether the data lies on a regularly spaced grid. Defaults to True.
        num_x (int, optional): The number of x-axis ticks. Defaults to 5.
        num_y (int, optional): The number of y-axis ticks. Defaults to 5.
        title (str, optional): The title of the plot. Defaults to "".
        x_label (str, optional): The label for the x-axis. Defaults to "".
        y_label (str, optional): The label for the y-axis. Defaults to "".

    Returns:
        None
    """
    fig, ax = plt.subplots()
    if regular:
        # Plot the 2D array
        im = ax.imshow(M, aspect="auto", origin="lower")
        ax.set_xticks(np.linspace(0, len(x), num_x))
        ax.set_yticks(np.linspace(0, len(y), num_y))
    else:
        # Create a 2D grid for the x and y coordinates
        X, Y = np.meshgrid(x, y)
        # Plot the 2D array
        im = ax.pcolormesh(X, Y, M)
        ax.set_xticks(np.linspace(x[0], x[-1], num_x))
        ax.set_yticks(np.linspace(y[0], y[-1], num_y))

    # Set the x and y axis labels and ticks
    ax.set_xticklabels(round_to_sig_figs(np.linspace(x[0], x[-1], num_x), 3))
    ax.set_yticklabels(round_to_sig_figs(np.linspace(y[0], y[-1], num_y), 3))
    # Set the title and labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label) 
    ax.set_title(title)
    # Add a colorbar
    plt.colorbar(im, ax=ax)
    
    return None