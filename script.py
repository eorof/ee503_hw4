# Gen by ChatGPT

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def read_data(filename):
    """Read numeric observations from a text file."""
    try:
        # Handles whitespace, tabs, and commas
        with open(filename, "r") as f:
            text = f.read().replace(",", " ")

        data = np.fromstring(text, sep=" ")

        if len(data) == 0:
            raise ValueError("No numeric data found.")

        return data

    except FileNotFoundError:
        print(f"Error: cannot find file '{filename}'")
        sys.exit(1)
    except ValueError as e:
        print(f"Error reading data: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python part2.py data.txt")
        sys.exit(1)

    filename = sys.argv[1]
    x = read_data(filename)

    n = len(x)

    # Sample statistics
    mean = np.mean(x)
    sd = np.std(x, ddof=1)  # sample SD, n-1 denominator

    print("===== DESCRIPTIVE STATISTICS =====")
    print(f"N                  = {n}")
    print(f"Sample mean        = {mean:.6f}")
    print(f"Sample SD          = {sd:.6f}")
    print()

    # ------------------------------------------------------------
    # 1. Histogram + fitted normal curve
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.hist(
        x,
        bins="auto",
        density=True,
        alpha=0.7,
        edgecolor="black",
        label="Observed data"
    )

    # Ideal normal curve using the sample mean and sample SD
    xmin = min(x)
    xmax = max(x)
    xx = np.linspace(xmin, xmax, 500)
    normal_pdf = stats.norm.pdf(xx, loc=mean, scale=sd)

    plt.plot(
        xx,
        normal_pdf,
        linewidth=2,
        label=f"Normal curve (mean={mean:.3f}, SD={sd:.3f})"
    )

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Histogram with Normal Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("histogram_normal.png", dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 2. P-P plot against normal distribution
    # ------------------------------------------------------------
    sorted_x = np.sort(x)

    # Empirical cumulative probabilities
    empirical_p = (np.arange(1, n + 1) - 0.5) / n

    # Theoretical cumulative probabilities under N(mean, sd)
    theoretical_p = stats.norm.cdf(
        sorted_x,
        loc=mean,
        scale=sd
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(
        theoretical_p,
        empirical_p,
        edgecolor="black"
    )

    # Perfect agreement line
    plt.plot(
        [0, 1],
        [0, 1],
        linewidth=2
    )

    plt.xlabel("Theoretical Normal CDF")
    plt.ylabel("Empirical CDF")
    plt.title("Normal P-P Plot")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pp_plot.png", dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 3. Shapiro-Wilk normality test
    # ------------------------------------------------------------
    shapiro_stat, shapiro_p = stats.shapiro(x)

    print("===== SHAPIRO-WILK NORMALITY TEST =====")
    print(f"W       = {shapiro_stat:.6f}")
    print(f"p-value = {shapiro_p:.6g}")

    if shapiro_p < 0.05:
        print("Result: Reject normality at alpha = 0.05")
    else:
        print("Result: Do not reject normality at alpha = 0.05")

    print()

    # ------------------------------------------------------------
    # 4. Anderson-Darling normality test
    # ------------------------------------------------------------
    ad = stats.anderson(x, dist="norm")

    print("===== ANDERSON-DARLING NORMALITY TEST =====")
    print(f"A^2 = {ad.statistic:.6f}")

    print("\nCritical values:")
    for significance, critical in zip(
        ad.significance_level,
        ad.critical_values
    ):
        print(
            f"{significance:.1f}% significance: "
            f"{critical:.6f}"
        )

    # ------------------------------------------------------------
    # Optional: D'Agostino-Pearson test
    # ------------------------------------------------------------
    if n >= 8:
        k2_stat, k2_p = stats.normaltest(x)

        print()
        print("===== D'AGOSTINO-PEARSON NORMALITY TEST =====")
        print(f"K^2     = {k2_stat:.6f}")
        print(f"p-value = {k2_p:.6g}")

        if k2_p < 0.05:
            print("Result: Reject normality at alpha = 0.05")
        else:
            print("Result: Do not reject normality at alpha = 0.05")

    print()
    print("Output files:")
    print("  histogram_normal.png")
    print("  pp_plot.png")


if __name__ == "__main__":
    main()
