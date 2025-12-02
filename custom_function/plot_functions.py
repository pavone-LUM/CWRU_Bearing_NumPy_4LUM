import matplotlib.pyplot as plt
import numpy as np
import math


def plot_signals(df, title="Signals"):
    """
    Automatically plots all signals in df['Raw'] with subplot layout
    and proper titles (df['Name']).
    """

    n = len(df)
    if n == 0:
        print("No data to plot.")
        return

    # Choose subplot grid size automatically
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        ax.plot(row["Raw"])
        ax.set_title(row["Name"], fontsize=10)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
