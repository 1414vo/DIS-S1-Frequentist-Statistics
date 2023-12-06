import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from src.distribution_utils.estimation import unbinned_mle_estimation


def plot_threshold_search(results, threshold):
    is_binary_search = len(results[0]) == 2
    x = np.array([res[0] for res in results[0]])
    y = np.array([res[1] for res in results[0]])
    y_err = np.sqrt(y / 1000)
    if not is_binary_search:
        hue = [abs(0.5 - res[2]) for res in results[0]]
    plt.errorbar(
        x, -np.log(y), yerr=y_err / y, ls="none", marker=None, ecolor="black", capsize=3
    )
    plt.axhline(-np.log(threshold), linestyle="--", label="T2 Error Threshold")
    plt.ylabel("-log(T2 Error)")
    plt.xlabel("Number of samples")
    if not is_binary_search:
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
        plt.axvline(results[1], label="Search Result", linestyle="--", color="red")
    plt.legend()
    plt.show()


def plot_t_statistic_distribution(
    h0_distribution, h1_distribution, t1_error_rate, t2_error_rate
):
    def chi2_pdf(X, df):
        return chi2.pdf(X, df=df)

    sample = h0_distribution[np.isfinite(h0_distribution)]
    chi2_fit, _, _ = unbinned_mle_estimation(
        sample, chi2_pdf, params={"df": 3}, limits={"df": (0, None)}
    )
    plt.hist(
        h0_distribution,
        bins=int(len(h0_distribution) ** 0.5),
        label="T statistic under H0",
        density=True,
    )
    plt.hist(
        h1_distribution,
        bins=int(len(h1_distribution) ** 0.5),
        label="T statistic under H1",
        density=True,
    )
    X = np.linspace(0, 2 * max(h0_distribution), 1000)
    plt.plot(X, chi2_pdf(X, chi2_fit["df"]), label="Chi-squared distribution fit")
    plt.axvline(
        chi2.ppf(1 - t1_error_rate, df=chi2_fit["df"]),
        label=f"$\alpha = {t1_error_rate}$",
    )
    plt.axvline(
        np.quantile(h1_distribution, t2_error_rate),
        label=f"$1 - \beta = {t2_error_rate}$",
    )
    plt.xlabel("Log-Likelihood Ratio")
    plt.ylabel("Test statistic probability density")
    plt.show()
