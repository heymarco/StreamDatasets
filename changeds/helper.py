import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_change_region_2d(regional_change_stream, change_idx: int, binary_thresh: float, save: bool, path=None):
        region = regional_change_stream.approximate_change_regions()[change_idx]
        region = (region - np.min(region)) / (np.max(region) - np.min(region))
        region = region.reshape(int(np.sqrt(len(region))), int(np.sqrt(len(region))))
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        for ax in axes:
            ax.set_aspect(1.0, adjustable='box')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        sns.heatmap(region, ax=axes[0], cmap="Greys", cbar_kws={"shrink": 0.5})
        sns.heatmap(region > (1-binary_thresh) * np.max(region), ax=axes[1], cmap="Greys_r", cbar_kws={"shrink": 0.5})
        axes[0].set_title("Mean difference")
        axes[1].set_title("Thresholded")
        plt.suptitle("Change for index {}".format(change_idx))
        plt.tight_layout()
        if save:
            assert isinstance(path, str)
            plt.savefig(path)
        else:
            plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])
