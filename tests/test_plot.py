import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdance import data
from mdance.outputs.plot import _covered_points, plot_n_clusters, plot_scores, read_scores, suggest_values


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def write_scores(path, header, rows):
    with open(path, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')
    return str(path)


def test_matches_previous_analysis():
    """Same DBI suggestions as the previous analysis notebooks (see test_db.py)."""
    x, ch, db, _ = read_scores(data.nani_screening)
    assert suggest_values(x, db, lower_is_better=True) == [6, 5, 8, 6]


@pytest.mark.parametrize('header, ch_col, db_col', [
    ('# NANI\n# Number of clusters, Calinski-Harabasz, Davies-Bouldin, Average MSD, Number of iterations\n', 1, 2),
    ('# eQual\n# Threshold, Number of clusters, Calinski-Harabasz, Davies-Bouldin, Average MSD\n', 2, 3),
    # Older headers, e.g. in the bundled nani_screening.csv
    ('# NANI\n# Number of clusters, Number of iterations, Calinski-Harabasz score, Davies-Bouldin score, Average MSD\n', 2, 3),
    ('# HELM\n# n_clusters,CH,DB\n', 1, 2),
    ('# DIVINE\n# n_clusters,CH_score,DB_score\n', 1, 2),
    ('# eQual\n# threshold,Calinski-Harabasz,Davies-Bouldin\n', 1, 2),
])
def test_read_scores_finds_columns(tmp_path, header, ch_col, db_col):
    """The CHI and DBI columns are found from the header of each method's scores file."""
    rows = []
    for k in range(2, 6):
        row = [k] + [0] * (header.count(',') + 1 - 1)
        row[ch_col], row[db_col] = 100 * k, 0.5 * k
        rows.append(row)
    x, ch, db, _ = read_scores(write_scores(tmp_path / 'scores.csv', header, rows))
    assert np.array_equal(x, [2, 3, 4, 5])
    assert np.array_equal(ch, [200, 300, 400, 500])
    assert np.array_equal(db, [1.0, 1.5, 2.0, 2.5])


def test_read_scores_skips_missing_rows(tmp_path):
    path = write_scores(tmp_path / 'scores.csv', '# n_clusters,CH,DB\n',
                        [(2, 'None', 'None'), (3, 10, 1.0), (4, 12, 0.8)])
    x, ch, db, _ = read_scores(path)
    assert np.array_equal(x, [3, 4])


def test_read_scores_needs_columns(tmp_path):
    path = write_scores(tmp_path / 'scores.csv', '# a,b,c\n', [(2, 1, 1), (3, 2, 2)])
    with pytest.raises(ValueError):
        read_scores(path)
    x, ch, db, _ = read_scores(path, ch_col=1, db_col=2)
    assert np.array_equal(ch, [1, 2])


def test_suggest_values_dbi_and_chi():
    x = np.arange(2, 11)
    dbi = np.array([1.5, 1.2, 1.4, 0.9, 1.3, 1.1, 1.6, 1.0, 1.2])
    # min 5, second min 9; local minima 3, 5, 7, 9 -> sharpest at 5 (0.9), then 9 (0.8)
    assert suggest_values(x, dbi, lower_is_better=True) == [5, 9, 5, 9]
    # CHI: higher is better, so the same values mirrored give the same suggestions
    assert suggest_values(x, -dbi, lower_is_better=False) == [5, 9, 5, 9]


def test_suggest_values_x_min_any_order():
    """x_min excludes small values whatever the row order (the previous DIVINE notebook kept them)."""
    x = np.arange(2, 11)
    dbi = np.array([0.1, 0.2, 1.4, 0.9, 1.3, 1.1, 1.6, 1.0, 1.2])
    ascending = suggest_values(x, dbi, True, x_min=5)
    descending = suggest_values(x[::-1], dbi[::-1], True, x_min=5)
    # x >= 5: min 5 (0.9), second min 9 (1.0); local minima 7 (sharpness 0.7) and 9 (0.8)
    assert ascending == descending == [5, 9, 9, 7]
    assert suggest_values(x, dbi, True)[0] == 2


def test_suggest_values_missing_entries():
    """Monotonic curves have no 2nd derivative suggestions."""
    assert suggest_values([2, 3, 4], [3.0, 2.0, 1.0], True) == [4, 3, None, None]
    assert suggest_values([2], [1.0], True) == [2, None, None, None]


def test_plot_scores_outputs(tmp_path):
    path = write_scores(tmp_path / 'equal_screening_scores.csv', '# eQual\n# threshold,Calinski-Harabasz,Davies-Bouldin\n',
                        [(5.0, 300, 2.0), (5.1, 320, 1.8), (5.2, 310, 1.9), (5.3, 330, 1.7)])
    suggested, figs = plot_scores(path)
    assert (tmp_path / 'equal_screening_scores_dbi.png').exists()
    assert (tmp_path / 'equal_screening_scores_chi.png').exists()
    assert list(suggested) == ['DBI', 'CHI']
    assert list(suggested['DBI']) == ['Min', 'Second Min', 'Max 2nd Deriv.', 'Second Max 2nd Deriv.']
    assert list(suggested['CHI']) == ['Max', 'Second Max', 'Min 2nd Deriv.', 'Second Min 2nd Deriv.']
    assert suggested['DBI']['Min'] == pytest.approx(5.3)
    assert len(figs) == 2
    assert figs[0].axes[0].get_xlabel() == 'Threshold'


def test_plot_scores_legend_placement(tmp_path):
    """The legend stays in the top right unless it covers the curve."""
    falling = write_scores(tmp_path / 'falling.csv', '# n_clusters,CH,DB\n',
                           [(k, 1000 / k, 2.0 / k) for k in range(2, 30)])
    rising = write_scores(tmp_path / 'rising.csv', '# n_clusters,CH,DB\n',
                          [(k, 10 * k, 0.1 * k) for k in range(2, 30)])
    _, falling_figs = plot_scores(falling)
    _, rising_figs = plot_scores(rising)
    upper_right = 1
    assert falling_figs[0].axes[0].get_legend()._loc == upper_right
    assert rising_figs[0].axes[0].get_legend()._loc != upper_right
    # The moved legend does not cover the curve either
    ax = rising_figs[0].axes[0]
    x, y = ax.lines[0].get_data()
    assert _covered_points(ax, ax.get_legend(), x, y) == 0


def test_plot_n_clusters(tmp_path):
    """Largest number of clusters and its first threshold, with rounded thresholds."""
    path = write_scores(tmp_path / 'equal_screening.csv', '# eQual\n# threshold,n_clusters,top_10_combined\n',
                        [(5.0, 12, 0.9), (5.1, 17, 0.9), (5.199999999999999, 17, 0.9), (5.299999999999999, 9, 0.9)])
    most, fig = plot_n_clusters(path)
    assert most == {'Max Number of Clusters': 17, 'Threshold': 5.1}
    assert (tmp_path / 'equal_screening_nclusters.png').exists()
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Threshold' and ax.get_ylabel() == 'Number of clusters'
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Max Number of Clusters: 17 (Threshold: 5.1)']
