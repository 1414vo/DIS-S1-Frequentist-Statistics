r"""! @file hypothesis_plots.py
@brief Contains methods for visualising the process of hypothesis testing and determining
the dataset size threshold.

@details Contains methods for visualising the process of hypothesis testing and determining
the dataset size threshold. The former is acheived by showing that as the dataset size increases,
the T2 error threshold moves to a greater value of the test statistic. The latter is done by plotting the
sampled values with their associated errors.

@author Created by I. Petrov on 06/12/2023
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, List, Dict
from scipy.stats import chi2
from src.distribution_utils.estimation import unbinned_mle_estimation
from sklearn.cluster import KMeans


def plot_threshold_search(
    results: Tuple[List, List, List], threshold: float, n_iter: int = 1000
) -> None:
    """! Visualizes the search algorithm for finding the dataset size corresponding to a T2 error threshold.

    @param results      A tuple of 1) A list of dataset sizes, 2) The corresponding T2 error rate,
    3) If using the weighted search algorithm, the Poisson p-values for each data point.
    @param threshold    The T2 error threshold.
    @param n_iter       The number of iterations done per hypothesis test."""

    plt.figure(figsize=(10, 6), dpi=150)

    # We have different elements for whether the search that was done was the binary search method
    is_binary_search = len(results[0][0]) == 2

    x = np.array([res[0] for res in results[0]])
    y = np.array([res[1] for res in results[0]])
    y_err = np.sqrt(y / n_iter)

    if not is_binary_search:
        # We use the Poisson p-values to color the samples from the weighted search algorithm
        hue = [abs(0.5 - res[2]) for res in results[0]]

    # Plot the search error bars and threshold
    plt.errorbar(
        x, -np.log(y), yerr=y_err / y, ls="none", marker=None, ecolor="black", capsize=3
    )
    plt.axhline(-np.log(threshold), linestyle="--", label="T2 Error Threshold")
    if not is_binary_search:
        # Plot the searched samples, colored by the Poisson p-value (closer to 0.5 is better)
        plt.scatter(
            x,
            -np.log(y),
            s=40,
            c=hue,
            cmap="YlOrBr",
            label="Search samples",
            zorder=2,
            marker="X",
        )
        # Display the found range.
        plt.axvline(
            results[1][1], label="Inferred lower bound", linestyle="--", color="red"
        )
        plt.axvline(
            results[1][0], label="Inferred upper bound", linestyle="--", color="red"
        )
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Deviation from the maximum likelihood point.")
    else:
        plt.scatter(x, -np.log(y), s=40, label="Search samples", zorder=2, marker="X")
        # Display the discovered value
        plt.axvline(results[1], label="Search Result", linestyle="--", color="red")

    plt.ylabel("-log(T2 Error)")
    plt.xlabel("Number of samples")
    plt.title("Distribution of T2 Errors of sampled dataset sizes")
    plt.tight_layout()
    plt.legend()
    plt.show()


def __select_samples(size_samples: np.ndarray, n_samples: int = 4) -> List[int]:
    r"""! From a set of dataset sizes, selects a number such that the pairwise distance between them is maximized.
    This is achieved through a K-means clustering algorithm, in which we take the closest point to each centroid.

    @param size_samples A NumPy array containing the sampled dataset sizes.
    @param n_samples    The number of samples we want to plot.

    @return             The datapoints corresponding to the maximum size discrimination.
    """

    # Fit a K-means model with the required number of clusters
    model = KMeans(n_clusters=n_samples, random_state=0).fit(
        size_samples.reshape(-1, 1)
    )
    centroids = model.cluster_centers_
    pred = model.predict(size_samples.reshape(-1, 1))

    relevant_samples = []

    # Take the closest points to each centroid
    for i in range(len(centroids)):
        points = size_samples[(pred == i).reshape(-1)]
        dist = abs(points - centroids[i])
        relevant_samples.append(points[np.argmin(dist)])

    return relevant_samples


def plot_t_statistic_distribution(
    h0_distribution: np.ndarray,
    h1_distributions: Dict[int, List[float]],
    t1_error_rate: float,
    t2_error_rate: float,
) -> None:
    """! Visualizes the obtained test statistic distributions after a search has been executed.

    @param h0_distribution  Data points obtained from generating the test statistic under the null hypothesis.
    @param h1_distributions Data points obtained from generating the test statistic under the alternate hypothesis.
    @param t1_error_rate    The Type 1 error threshold.
    @param t2_error_rate    The Type 2 error threshold."""
    plt.figure(figsize=(10, 6), dpi=150)

    # Fit a Chi-squared distribution for the H0 distribution
    def chi2_pdf(X, df):
        return chi2.pdf(X, df=df)

    sample = h0_distribution[np.isfinite(h0_distribution)]
    chi2_fit, _, _ = unbinned_mle_estimation(
        sample, chi2_pdf, params={"df": 3}, limits={"df": (0, None)}
    )
    # Plot the Null Hypothesis distribution
    plt.hist(
        h0_distribution,
        bins=int(len(h0_distribution) ** 0.5),
        label="T statistic under H0",
        density=True,
    )

    # The Chi-squared fit
    X = np.linspace(0, 2 * max(h0_distribution), 1000)
    plt.plot(X, chi2_pdf(X, chi2_fit["df"]), label="Chi-squared distribution fit")

    # And the relevant T2 error threshold
    plt.axvline(
        chi2.ppf(1 - t1_error_rate, df=chi2_fit["df"]),
        label=rf"$\alpha = {t1_error_rate}$",
        linestyle="--",
        color="red",
    )

    # Get 4 discriminative samples using K-means clustering
    # Method is not relevant to the report, it is an easy to set up heuristic
    # So that similar sample sizes are not plotted together
    samples = np.fromiter(h1_distributions.keys(), dtype=float)
    relevant_samples = __select_samples(samples)
    for n_samples in sorted(relevant_samples):
        # For each sample, plot the distribution...
        h = plt.hist(
            h1_distributions[n_samples],
            bins=int(len(h1_distributions[n_samples]) ** 0.5),
            label=f"T statistic under H1 for {int(n_samples)} data points",
            density=True,
            alpha=0.5,
        )
        # And the T1 error threshold
        plt.axvline(
            np.quantile(h1_distributions[n_samples], t2_error_rate),
            label=rf"$1 - \beta = {t2_error_rate}$ for {int(n_samples)} data points",
            linestyle="--",
            color=h[2][0].get_facecolor(),
            alpha=1,
        )

    plt.xlabel("Log-Likelihood Ratio")
    plt.xlim(0, 120)
    plt.ylim(0, 0.4)
    plt.ylabel("Test statistic probability density")
    plt.title("Test statistic distributions for different dataset sizes")
    plt.legend(bbox_to_anchor=(1, 1, 0, 0.0))
    plt.tight_layout()
    plt.show()
