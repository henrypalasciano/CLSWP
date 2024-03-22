import numpy as np
import matplotlib.pyplot as plt

def view(M: np.ndarray, x: np.ndarray, y: np.ndarray, num_x: int = 5, 
         num_y: int = 5, title: str = "", x_label: str = "", y_label: str = "") -> None:
    """
    Display a 2D array as an image with labeled axes and a colorbar.

    Args:
        M (np.ndarray): The 2D array to be displayed.
        x (np.ndarray): The x-axis values.
        y (np.ndarray): The y-axis values.
        num_x (int, optional): The number of x-axis ticks. Defaults to 5.
        num_y (int, optional): The number of y-axis ticks. Defaults to 5.
        title (str, optional): The title of the plot. Defaults to "".
        x_label (str, optional): The label for the x-axis. Defaults to "".
        y_label (str, optional): The label for the y-axis. Defaults to "".

    Returns:
        None
    """
    # Plot the 2D array as an image
    fig, ax = plt.subplots()
    im = ax.imshow(M, aspect="auto", origin="lower")
    # Set the x and y axis labels and ticks
    ax.set_xticks(np.linspace(0, len(x), num_x))
    ax.set_xticklabels(np.linspace(x[0], x[-1], num_x))
    ax.set_yticks(np.linspace(0, len(y), num_y))
    ax.set_yticklabels(np.linspace(y[0], y[-1], num_y))
    # Set the title and labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label) 
    ax.set_title(title)
    # Add a colorbar
    plt.colorbar(im, ax=ax)
    
    return None