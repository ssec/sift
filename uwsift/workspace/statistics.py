import logging

import numpy as np

LOG = logging.getLogger(__name__)


def dataset_statistical_analysis(xarr):
    """Compute and return a dictionary with statistical information about the input dataset.

    The dataset should be of type xarray.DataArray (usually Satpy Scene objects) such that the dataset attributes
    can be used to compute and return the appropriate statistical information.
    """

    stats = None
    if "flag_values" in xarr.attrs:
        # Categorical data
        flag_values = xarr.attrs["flag_values"]

        try:
            flag_meanings = xarr.attrs["flag_meanings"]
        except (KeyError, AttributeError):
            LOG.debug("'flag_meanings' not available as dataset attributes, setting to n/a.")
            flag_meanings = ["n/a"] * len(flag_values)

        stats = CategoricalBasicStats(flag_values, flag_meanings)

    elif "flag_masks" in xarr.attrs:
        # Bit-encoded data
        return {}

    elif "algebraic" in xarr.attrs:
        # Algebraic data. At least for differences, we want to have some additional statistical metrics.
        normalized_formula = xarr.attrs["algebraic"].replace(" ", "")
        if normalized_formula == "x-y" or normalized_formula == "y-x":
            stats = ContinuousDifferenceStats()
        else:
            LOG.debug(f"'ContinuousBasicStats' will be computed for algebraic operation {xarr.attrs['algebraic']}.")

    elif xarr.dtype.kind == "i" and len(np.unique(xarr)) < 25:
        # NOTE: This is a preliminary ugly workaround used to identify categorical dataset and guess the categories.
        # TODO: Modify satpy readers to provide proper information using flag_values and flag_meanings attributes.
        flag_values = np.arange(0, np.nanmax(xarr) + 1)
        flag_meanings = ["n/a"] * len(flag_values)
        stats = CategoricalBasicStats(flag_values, flag_meanings)

    # All remaining datasets, including basic continuous data.
    if not stats:
        stats = ContinuousBasicStats()

    stats.compute_stats(xarr.values)
    stats_out = stats.get_stats()

    return stats_out


class ContinuousBasicStats:
    """Basic statistical metrics to use for continuous datasets."""

    def __init__(self):
        self.stats = {
            "count": [],
            "min": [],
            "max": [],
            "mean": [],
            "median": [],
            "std": [],
        }

    def compute_stats(self, data):
        self.compute_basic_stats(data)

    def compute_basic_stats(self, data):
        self.stats["count"].append(np.count_nonzero(~np.isnan(data)))
        self.stats["min"].append(np.nanmin(data))
        self.stats["max"].append(np.nanmax(data))
        self.stats["mean"].append(np.nanmean(data))
        self.stats["median"].append(np.nanmedian(data))
        self.stats["std"].append(np.nanstd(data))

    def get_stats(self):
        """Send the statistical data to a statistics dictionary.

        The output dictionary shall have the following format::

            stats_dict = {
                stats: {
                    'statistical_metric_i': [statistical_value_i],
                    'statistical_metric_j': [statistical_value_j],
                    'statistical_metric_k': [statistical_value_k],
                }
            }

        where i, j, k represents the different statistical metrics.
        """
        stats_dict = {"stats": self.stats}
        return stats_dict


class ContinuousDifferenceStats(ContinuousBasicStats):
    """Statistical metrics to use for continuous difference datasets."""

    def __init__(self):
        super().__init__()  # initialize basic continuous statistics
        self.stats["mad"] = []
        self.stats["rmsd"] = []

    def compute_stats(self, diff):
        self.compute_basic_stats(diff)
        self.compute_difference_stats(diff)

    def compute_difference_stats(self, diff):
        """Compute additional statistical metrics useful for difference datasets."""
        self.stats["mad"].append(np.nanmean(np.abs(diff)))
        self.stats["rmsd"].append(np.sqrt(np.nanmean(np.square(diff))))


class CategoricalBasicStats:
    """Basic statistical metrics to use for categorical datasets."""

    def __init__(self, flag_values, flag_meanings):
        self.header = ["value", "meaning", "count / -", "fraction / %"]
        self.flag_values = list(flag_values)
        self.flag_meanings = list(flag_meanings)
        self.count = []
        self.fraction = []

    def compute_stats(self, data):
        self.compute_basic_stats(data)

    def compute_basic_stats(self, data):
        """Compute the number and fraction (wrt. total count) of a given category."""
        self.count = [np.count_nonzero(data == val) for val in self.flag_values]
        self.fraction = [(c / sum(self.count) * 100.0 if sum(self.count) > 0.0 else 0.0) for c in self.count]

    def get_stats(self):
        """Put the statistical data in a list of lists and send together with header to a statistics dictionary.

        The output dictionary shall have the following format::

            stats_dict = {
                header: ['value', 'meaning', 'count / -', 'fraction / %']
                stats: [
                    [value_i, meaning_i, count_i, fraction_i],
                    [value_j, meaning_j, count_j, fraction_j],
                    [value_k, meaning_k, count_k, fraction_k],
                ]
            }

        where i, j, k represents the values representing the different categories.
        """
        stats = [
            list((value, meaning, count, fraction))
            for value, meaning, count, fraction in zip(self.flag_values, self.flag_meanings, self.count, self.fraction)
        ]
        stats_dict = {
            "header": self.header,
            "stats": stats,
        }

        return stats_dict
