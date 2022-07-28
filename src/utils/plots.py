import matplotlib.pyplot as plt
import pyautogui

from src.settings import settings
import matplotlib


def init_plots(nb_rows=2, nb_cols=2):
    plt.ion()
    # PLOTS INITIALISATION

    fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols,
                            figsize=(settings.plots_window_width / 100, settings.plots_window_height / 100))
    # X, y  stuff to plot
    # Top left:     Goals buffer coverage, representation of states inside goals buffer
    # Top right:    Agent training accuracy, accuracy of the agent on goals inside their buffer
    # Bottom left:  Goals sampled for tests (we plot the last test ones)
    # Bottom right: Agent testing accuracy, average accuracy on test goals

    # Scale and place window to make it more comfortable.
    plt.subplots_adjust(hspace=0.4)  # Set sub plot margin
    # Then place window on the screen:
    width, height = pyautogui.size()

    move_figure(fig, (width - int(settings.plots_window_width)) // 2,
                (height - int(settings.plots_window_height)) // 2 - 30)

    return fig, axs


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
