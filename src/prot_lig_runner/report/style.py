"""Figure style shared by every analysis plot (Nature-compatible defaults).

Single source of truth for fonts, line widths, palette and export settings, so
the figures in the report follow the same rules as publication figures: no
titles, 7 pt text, 0.5 pt spines, Wong palette, 450 dpi PNG.
"""

from __future__ import annotations

PALETTE = ["#0072b2", "#d55e00", "#009e73", "#e69f00", "#cc79a7", "#56b4e9", "#000000", "#f0e442"]


def apply_style():
    """Install rcParams for report figures; safe to call repeatedly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 2.6,
        "legend.frameon": False,
        "legend.borderaxespad": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def moving_average(values, window):
    """Centred moving average that keeps the array length; NaN outside edges."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    window = int(window)
    if window <= 1 or values.size < window:
        return values.copy()
    kernel = np.ones(window)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = sums / counts
    return np.where(counts > 0, smoothed, np.nan)


def smooth_with_spread(ax, x, y, *, window, colour, label, band=True, raw=True):
    """Raw trace in the background, smoothed line on top, ±SD band around it."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if raw:
        ax.plot(x, y, color=colour, linewidth=0.35, alpha=0.28, label="_nolegend_")
    if window > 1 and y.size > window:
        smooth = moving_average(y, window)
        residual = y - smooth
        spread = moving_average(residual ** 2, window) ** 0.5
        if band:
            ax.fill_between(x, smooth - spread, smooth + spread, color=colour, alpha=0.18,
                            linewidth=0)
        ax.plot(x, smooth, color=colour, linewidth=0.9, label=label)
    else:
        ax.plot(x, y, color=colour, linewidth=0.9, label=label)
