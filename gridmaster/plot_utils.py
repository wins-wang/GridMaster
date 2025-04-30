import matplotlib.pyplot as plt

__PLOT_STYLE_APPLIED__ = False

def set_plot_style(font_size=14, force=False):
    """
    Set default matplotlib plot style for readable visuals.

    This function sets global font sizes and styling parameters for all plots,
    improving readability especially in notebooks or high-resolution screens.
    It only applies once per session unless force=True.

    Args:
        font_size (int): Base font size. Defaults to 14.
        force (bool): If True, force re-apply even if already set. Defaults to False.

    Returns:
        None
    """
    global __PLOT_STYLE_APPLIED__
    if __PLOT_STYLE_APPLIED__ and not force:
        return
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2,
    })
    __PLOT_STYLE_APPLIED__ = True
