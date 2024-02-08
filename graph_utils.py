"""
This module contains utilities for EDA
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from distribution_utils import is_gaussian


def bar_chart(df:pd.DataFrame, var: str, out: bool = False) -> None:
    """
    Draw bar chart for nominal variables
    `out` set to True allows to export the graph in png format in the graphs folder.
    """
    total = len(df)
    ax = sns.countplot(x=var, data=df,
                       hue=var,
                       legend=False)
    
    for p in ax.patches:
        height = p.get_height()
        percentage = '{:.0f}%'.format(100 * height/total)
        ax.text(p.get_x() + p.get_width() / 2., height + 3, percentage, ha="center")
        ax.set_title(f"{var}'s bar chart",
                     size="x-large", weight="bold", color="blue")
        
        plt.xticks(rotation=90)

        # Customize labels on x and y axis
        plt.xlabel(var, fontsize=11, fontweight="bold")
        plt.ylabel("count", fontsize=11, fontweight="bold")

    if out:
        plt.savefig(f"graphs/{var}_bar_chart.png", dpi=300)
    
    plt.show()


def contingency_table(
    var1: str, var2: str, df: pd.DataFrame,
    out: bool = False) -> pd.DataFrame:
    """
    Get the contingency table for two nominal features of a DataFrame
    `out` set to True allows to export the output in .csv format in csvs folder.
    """
    ct = pd.crosstab(df[var1], df[var2])
    if out:
        ct.to_csv(f"csvs/{var1}_{var2}_ct.csv", index=True)
    return ct


def correlation_heatmap(df: pd.DataFrame, out: bool = False) -> None:
    """
    Displays correlations between numerical features of a DataFrame.
    `out` set to True allows to export the graph in png format in the graphs folder.
    """
    corr = df.select_dtypes(include="number").corr()
    # Create a matrix full of zeros similar to corr
    mask = np.zeros_like(corr)
    # Set the upper triangle of the mask to True
    mask[np.triu_indices_from(mask)] = True
    
    with sns.axes_style("white"):
      fig, ax = plt.subplots(figsize=(6, 6))
      ax = sns.heatmap(
        corr,
        # MASK
        mask=mask,
        # Div palette, suffix _r to reverse
        cmap="RdBu_r", # coolwarm, vlag, icefire
        # Text in heatmap
        annot=True,# Allowing annotations.
        fmt=".2f", # Formatting annotations. 
        annot_kws=dict(
          fontsize=9,
          fontweight="bold"
          ), # Other formatting
        # Values on vertical colorbar
        vmax=1,
        center=0,
        vmin=-1,
        )
    plt.title("Correlations Heatmap", size="xx-large", weight="bold", c="b")

    if out:
        plt.savefig("graphs/correlations_heatmap.png", dpi=300)
        
    plt.show()

    
def hist_box_plot(df: pd.DataFrame, col: str, out: bool = False) -> None:
    """
    Display an histogram and a boxplot of df[col],
    col being a continuous numerical variable.
    `out` set to True allows to export the graph in .png format.
    """
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)})
    mean=df[col].mean()
    median=df[col].median()
    mode=df[col].mode().values[0]
    isGaussian = is_gaussian(df, col)
     
    sns.boxplot(data=df, x=col, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    ax_box.axvline(mode, color='b', linestyle='-')
     
    sns.histplot(data=df, x=col, ax=ax_hist, kde=True)
    ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
    ax_hist.axvline(median, color='g', linestyle='-', label="Median")
    ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
     
    ax_hist.legend(loc='best')
     
    # Set x-axis label
    ax_hist.set_xlabel(col, fontsize=11, fontweight='bold')

    # Set y-axis label for the histogram
    ax_hist.set_ylabel('Count', fontsize=11, fontweight='bold')

    # Clear the x-axis label for the boxplot (as they share the same x-axis)
    ax_box.set(xlabel='')

    plt.suptitle(f"Distribution of {col}",
                 size="x-large", weight="bold", color="blue")

    plt.title(f'{"normale" if isGaussian else "non normale"}')
    
    if out:
        plt.savefig(f"graphs/{col}_hist_box.png", dpi=300)
        
    
    plt.show()
