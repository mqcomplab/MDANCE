import matplotlib.pyplot as plt
import numpy as np


LINE_COLOR = '#0072B2'
MARK_COLORS = ['#D55E00', '#009E73', '#CC79A7', '#56B4E9']
MARK_STYLES = ['--', '--', ':', ':']


def read_scores(scores_csv, ch_col=None, db_col=None):
    """Reads a screening scores file and finds the CHI and DBI columns.

    Parameters
    ----------
    scores_csv : str
        Path to the scores file. The first column is the screened value
        (number of clusters or threshold). Lines starting with ``#`` are
        comments; the last one names the columns.
    ch_col : int, default=None
        Column of the Calinski-Harabasz index. Found from the header if None.
    db_col : int, default=None
        Column of the Davies-Bouldin index. Found from the header if None.

    Returns
    -------
    tuple
        ``(x, ch, db, x_name)``. Rows with missing scores are removed.

    Raises
    ------
    ValueError
        If a column is not given and cannot be found in the header.
    """
    with open(scores_csv) as f:
        header = [line[1:].strip() for line in f if line.startswith('#')]
    names = [name.strip() for name in header[-1].split(',')] if header else []
    if ch_col is None:
        ch_col = _find_column(names, ('Calinski', 'CH'), 'Calinski-Harabasz')
    if db_col is None:
        db_col = _find_column(names, ('Davies', 'DB'), 'Davies-Bouldin')
    data = np.genfromtxt(scores_csv, delimiter=',', comments='#', usecols=(0, ch_col, db_col))
    data = data[~np.isnan(data).any(axis=1)]
    x_name = names[0] if names else ''
    return data[:, 0], data[:, 1], data[:, 2], x_name


def _find_column(names, prefixes, index_name):
    for i, name in enumerate(names):
        if name.startswith(prefixes):
            return i
    raise ValueError(f'Could not find the {index_name} column in the header {names}. '
                     f'Pass its column number explicitly.')


def suggest_values(x, y, lower_is_better, x_min=None):
    """Suggests values of ``x`` from a clustering quality index.

    Parameters
    ----------
    x : array-like
        Screened values (number of clusters or threshold).
    y : array-like
        Index value for each ``x``.
    lower_is_better : bool
        True for the Davies-Bouldin index, False for the Calinski-Harabasz index.
    x_min : float, default=None
        Only values of ``x`` greater than or equal to ``x_min`` are suggested.

    Returns
    -------
    list
        ``[best, second best, best 2nd derivative, second best 2nd derivative]``.
        An entry is None if it does not exist.

    Notes
    -----
    The best values are the minimum (DBI) or maximum (CHI) of the index. The 2nd
    derivative suggestions are the local minima (DBI) or maxima (CHI) with the
    sharpest change, ``y[i+1] + y[i-1] - 2*y[i]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    if x_min is not None:
        keep = x >= x_min
        x, y = x[keep], y[keep]
    s = y if lower_is_better else -y
    best = list(x[np.argsort(s, kind='stable')][:2])
    knees = []
    for i in range(1, len(s) - 1):
        if s[i] <= s[i - 1] and s[i] <= s[i + 1]:
            knees.append((s[i + 1] + s[i - 1] - 2 * s[i], x[i]))
    knees = [knee for _, knee in sorted(knees, key=lambda k: -k[0])][:2]
    values = (best + [None, None])[:2] + (knees + [None, None])[:2]
    return [_as_number(v) for v in values]


def _as_number(value):
    if value is None:
        return None
    value = round(float(value), 10)     # e.g. thresholds from np.arange: 5.399999999999999 -> 5.4
    return int(value) if value.is_integer() else value


def plot_scores(scores_csv, x_min=None, ch_col=None, db_col=None, xlabel=None,
                output_prefix=None, dpi=200):
    """Plots the Davies-Bouldin and Calinski-Harabasz indices of a screening
    and suggests the number of clusters (or threshold).

    Parameters
    ----------
    scores_csv : str
        Path to the screening scores file, e.g. ``outputs/helm_intra_scores.csv``.
    x_min : float, default=None
        Only values greater than or equal to ``x_min`` are suggested. The whole
        screening is still plotted.
    ch_col : int, default=None
        Column of the Calinski-Harabasz index. Found from the header if None.
    db_col : int, default=None
        Column of the Davies-Bouldin index. Found from the header if None.
    xlabel : str, default=None
        Label of the x-axis. ``Threshold`` if the first column is a threshold,
        otherwise ``Number of clusters``.
    output_prefix : str, default=None
        The figures are saved as ``{output_prefix}_dbi.png`` and
        ``{output_prefix}_chi.png``. Defaults to ``scores_csv`` without ``.csv``.
    dpi : int, default=200
        Resolution of the saved figures.

    Returns
    -------
    dict
        Suggested values. ``{'DBI': {'Min': ..., 'Second Min': ...,
        'Max 2nd Deriv.': ..., 'Second Max 2nd Deriv.': ...}, 'CHI': {...}}``
    tuple
        The DBI and CHI figures.

    Examples
    --------
    >>> from mdance.outputs.plot import plot_scores
    >>> suggested, figs = plot_scores('outputs/helm_intra_scores.csv', x_min=5)  # doctest: +SKIP
    >>> suggested['DBI']['Min']  # doctest: +SKIP
    """
    scores_csv = str(scores_csv)
    x, ch, db, x_name = read_scores(scores_csv, ch_col=ch_col, db_col=db_col)
    if xlabel is None:
        xlabel = 'Threshold' if 'threshold' in x_name.lower() else 'Number of clusters'
    if output_prefix is None:
        output_prefix = scores_csv[:-4] if scores_csv.endswith('.csv') else scores_csv

    suggested = {}
    figs = []
    for abbrev, full_name, y, lower_is_better in [('DBI', 'Davies-Bouldin Index', db, True),
                                                  ('CHI', 'Calinski-Harabasz Index', ch, False)]:
        best, deriv = ('Min', 'Max') if lower_is_better else ('Max', 'Min')
        labels = [best, f'Second {best}', f'{deriv} 2nd Deriv.', f'Second {deriv} 2nd Deriv.']
        values = suggest_values(x, y, lower_is_better, x_min=x_min)
        suggested[abbrev] = dict(zip(labels, values))
        marks = [(f'{label} {abbrev}: {value:g}' if value is not None else None, value)
                 for label, value in zip(labels, values)]
        fig = _plot_index(x, y, full_name, xlabel, marks)
        fig.savefig(f'{output_prefix}_{abbrev.lower()}.png', dpi=dpi)
        figs.append(fig)
    return suggested, tuple(figs)


def plot_n_clusters(screening_csv, output_prefix=None, dpi=200):
    """Plots the number of clusters found at each threshold of a screening.

    Parameters
    ----------
    screening_csv : str
        Path to the screening file, e.g. ``outputs/equal_screening.csv``. The
        first column is the threshold and the second is the number of clusters.
    output_prefix : str, default=None
        The figure is saved as ``{output_prefix}_nclusters.png``. Defaults to
        ``screening_csv`` without ``.csv``.
    dpi : int, default=200
        Resolution of the saved figure.

    Returns
    -------
    dict
        ``{'Max Number of Clusters': ..., 'Threshold': ...}``, the largest number
        of clusters and the (first) threshold where it is found.
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> from mdance.outputs.plot import plot_n_clusters
    >>> most, fig = plot_n_clusters('outputs/equal_screening.csv')  # doctest: +SKIP
    """
    screening_csv = str(screening_csv)
    data = np.genfromtxt(screening_csv, delimiter=',', comments='#', usecols=(0, 1), ndmin=2)
    data = data[~np.isnan(data).any(axis=1)]
    threshold, n_clusters = data[:, 0], data[:, 1]
    if output_prefix is None:
        output_prefix = screening_csv[:-4] if screening_csv.endswith('.csv') else screening_csv

    i = int(np.argmax(n_clusters))
    most = {'Max Number of Clusters': _as_number(n_clusters[i]), 'Threshold': _as_number(threshold[i])}
    label = f'Max Number of Clusters: {most["Max Number of Clusters"]:g} (Threshold: {most["Threshold"]:g})'
    fig = _plot_index(threshold, n_clusters, 'Number of clusters', 'Threshold', [(label, most['Threshold'])])
    fig.savefig(f'{output_prefix}_nclusters.png', dpi=dpi)
    return most, fig


def _plot_index(x, y, ylabel, xlabel, marks):
    order = np.argsort(x, kind='stable')
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(x, y, color=LINE_COLOR, linewidth=1.2, marker='o', markersize=2.2)
    for (label, value), color, style in zip(marks, MARK_COLORS, MARK_STYLES):
        if value is not None:
            ax.axvline(value, color=color, linestyle=style, linewidth=1.2, label=label)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7, width=0.8, length=3)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_zorder(10)
    fig.tight_layout()

    # Legend in the top right, unless it covers the curve: then the first free spot,
    # otherwise in the top right with extra room above the curve
    style = dict(fontsize=6, handlelength=1.6, labelspacing=0.3, borderaxespad=0, frameon=True,
                 facecolor='white', edgecolor='none', framealpha=1)
    for loc in ['upper right', 'upper left', 'lower right', 'lower left', 'upper center',
                'lower center', 'center right', 'center left', 'center']:
        legend = ax.legend(loc=loc, **style)
        if _covered_points(ax, legend, x, y) == 0:
            return fig
        legend.remove()
    bottom, top = ax.get_ylim()
    legend = ax.legend(loc='upper right', **style)
    for extra in np.arange(0.1, 2.01, 0.1):
        ax.set_ylim(bottom, top + extra * (top - bottom))
        if _covered_points(ax, legend, x, y) == 0:
            break
    return fig


def _covered_points(ax, legend, x, y):
    fig = ax.figure
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    t = np.linspace(0, 1, 25)
    xs = np.concatenate([x[:1]] + [x[i] + t * (x[i + 1] - x[i]) for i in range(len(x) - 1)])
    ys = np.concatenate([y[:1]] + [y[i] + t * (y[i + 1] - y[i]) for i in range(len(y) - 1)])
    points = ax.transData.transform(np.column_stack([xs, ys]))
    inside = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1) &
              (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1))
    return int(inside.sum())
