import matplotlib.pyplot as plt

plt.style.use("ggplot")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.sans-serif": "Computer Modern",
        # "text.latex.preamble": [r"\usepackage[varvw]{newtxmath}"],
    }
)


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional "aaai"
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "aaai":
        width = 239.39438
    elif width == "springer":
        width = 332.89723
    elif width == "memoire":
        width == 443
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
