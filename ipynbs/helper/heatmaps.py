from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap_from_df(df : pd.DataFrame , x=6, y=6):
    plt.figure(figsize=(x, y))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="mako")
    plt.show()