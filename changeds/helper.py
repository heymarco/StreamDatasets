import os.path
from tqdm import tqdm

import numpy as np
import pandas as pd
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


def preprocess_hipe() -> pd.DataFrame:
    this_dir, _ = os.path.split(__file__)
    folder = os.path.join(this_dir, "..", "data", "hipe")
    cache_dir = os.path.join(folder, "cache")
    cache_df_path = os.path.join(cache_dir, "df.csv")
    if os.path.exists(cache_df_path):
        data = pd.read_csv(cache_df_path)
        data["SensorDateTime"] = pd.to_datetime(data["SensorDateTime"], utc=True)
    else:
        dfs = []
        all_files = os.listdir(folder)
        for i, file in tqdm(enumerate(all_files)):
            if file.endswith(".csv"):
                path = os.path.join(folder, file)
                df = pd.read_csv(path)
                df["SensorDateTime"] = pd.to_datetime(df["SensorDateTime"], utc=True).dt.round("s")
                dfs.append(df)
        data = pd.concat(dfs, join="outer", ignore_index=True)
        data = data.sort_values("SensorDateTime").groupby("SensorDateTime").mean()
        data.to_csv(cache_df_path)
    data.ffill(inplace=True)  # forward fill if possible
    data.bfill(inplace=True)  # backward fill the rest
    return data.drop(["PhaseCount"], axis=1).to_numpy()



