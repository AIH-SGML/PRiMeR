import numpy as np
import matplotlib.pyplot as plt


def history_plot(history: dict, figname: str, show: bool = False):

    # make plot
    plt.figure(1, figsize=(10, 8))

    n = np.ceil(np.sqrt(len(history))).astype(int)
    for i, (k, v) in enumerate(history.items()):
        plt.subplot(n, n, i + 1)
        plt.plot(v)
        plt.title(k)

    plt.tight_layout()
    # dump it

    if show:
        plt.show()
    else:
        plt.savefig(figname, dpi=300, bbox_inches="tight")
        plt.close()
