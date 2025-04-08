#!/usr/bin/env python3
"""
analyze_extracted_fields_with_stacked_missing.py

Purpose:
  1) Creates vertical and horizontal bar plots for categorical fields,
     excluding "Not Inferred" entries so the main distribution is uncluttered.
  2) Additionally, for each field we generate a SINGLE "stacked bar" that
     displays how many rows are Inferred vs Not Inferred (missing).

Implementation Details:
  - Uses integer positions on bars to avoid TypeErrors.
  - Applies a minimum of 3 discrete color steps so small category sets
    do not become too faint.
  - Large fonts, dpi=600.
  - The "stacked bar" is one bar per field: bottom = count(inferred),
    stacked portion = count(not_inferred).

Usage:
  python analyze_extracted_fields_with_stacked_missing.py \
    --path_csv "/Data/.../path_fields.csv" \
    --cons_csv "/Data/.../cons_fields.csv" \
    --out_dir  "/Data/.../Structured_Fields_plots/Exp14"
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


###############################################################################
# 1) Field to Colormap Mapping
###############################################################################
FIELD_COLORMAP = {
    "Alcohol_Consumption":          "Reds",
    "Smoking_History":              "Reds",
    "ECOG_Performance_Status":      "Greens",
    "Karnofsky_Performance_Status": "Greens",
    "Lymph_Node_Status_Presence_Absence": "Purples",
    # Fields not in this dict default to "Blues"
}

###############################################################################
# 2) Utility Functions
###############################################################################
def pretty_name(field_name):
    """
    Replace underscores with spaces for nicer plot titles/labels.
    E.g., "ECOG_Performance_Status" => "ECOG Performance Status"
    """
    return field_name.replace("_", " ")

def get_colormap_for_field(field_name):
    """Return the colormap string for the given field; default = 'Blues'."""
    return FIELD_COLORMAP.get(field_name, "Blues")

def get_color_list(n, cmap_name="Blues"):
    """
    Create a list of 'n' distinct colors from the given colormap.
    We set a minimum of 3 discrete steps so that if n is small,
    we don't end up with extremely faint bars.
    """
    n_use = max(n, 3)
    cmap = cm.get_cmap(cmap_name, n_use)
    return [cmap(i) for i in range(n_use)]

def exclude_not_inferred(series):
    """
    Return a filtered series that excludes rows with "Not Inferred" (case-insensitive),
    and also excludes raw NaN values.
    """
    if series.isna().all():
        return series.dropna()

    s_str = series.astype(str).str.strip().str.lower()
    mask = (s_str != "not inferred") & (s_str != "nan")
    return series[mask]

###############################################################################
# 3) Bar Plot Functions (Vertical & Horizontal, excluding "Not Inferred")
###############################################################################
def make_bar_plot_vertical(series, field_name, out_dir):
    """
    Vertical bar plot, excluding "Not Inferred".
    Positions are integer, categories become xticks.
    """
    valid_series = exclude_not_inferred(series)
    counts = valid_series.value_counts()
    if counts.empty:
        return  # no valid data to plot

    x_positions = range(len(counts))

    colormap_name = get_colormap_for_field(field_name)
    color_list = get_color_list(len(counts), colormap_name)

    plt.figure(figsize=(8, 5))
    plt.bar(x_positions, counts.values, color=color_list, width=0.8)
    plt.title(f"{pretty_name(field_name)} (Vertical Bar=", fontsize=20)
    plt.xlabel(pretty_name(field_name), fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(x_positions, [str(c) for c in counts.index],
               rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{field_name}_bar_vertical.png")
    plt.savefig(out_path, dpi=600)
    plt.close()

def make_bar_plot_horizontal(series, field_name, out_dir):
    """
    Horizontal bar plot, excluding "Not Inferred".
    Positions are integer, categories become yticks.
    Sort ascending => largest bar at the top if you prefer.
    """
    valid_series = exclude_not_inferred(series)
    counts = valid_series.value_counts()
    if counts.empty:
        return

    # Sort ascending => largest bar on top
    counts = counts.sort_values(ascending=True)

    y_positions = range(len(counts))
    colormap_name = get_colormap_for_field(field_name)
    color_list = get_color_list(len(counts), colormap_name)

    plt.figure(figsize=(8, 5))
    plt.barh(y_positions, counts.values, color=color_list)
    plt.title(f"{pretty_name(field_name)}", fontsize=20)
    plt.xlabel("Count", fontsize=18)
    plt.ylabel(pretty_name(field_name), fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(y_positions, [str(c) for c in counts.index], fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{field_name}_bar_horizontal.png")
    plt.savefig(out_path, dpi=600)
    plt.close()

###############################################################################
# 4) Single "Stacked" Bar for each field, showing Inferred vs Not Inferred
###############################################################################
def make_single_stacked_bar_inferred(series, field_name, out_dir):
    """
    Creates a SINGLE stacked bar for the entire field, with two segments:
      - Inferred Count  = # of rows that are not "Not Inferred"
      - Not Inferred    = # of rows that are "Not Inferred"
    """
    total_count = len(series)
    # Convert to string to detect "Not Inferred" ignoring case
    s_str = series.astype(str).str.strip().str.lower()
    not_inferred_count = (s_str == "not inferred").sum()
    inferred_count = total_count - not_inferred_count

    # If total_count == 0, do nothing
    if total_count == 0:
        return

    # We'll do a single bar at x=0, with 2 segments stacked
    # We'll color them differently so we can see the difference
    # e.g. [Inferred, Not Inferred]
    # colormap for the field
    colormap_name = get_colormap_for_field(field_name)
    cmap = cm.get_cmap(colormap_name, 2)  # just 2 steps
    color_inferred = cmap(0)
    color_not_inferred = cmap(1)

    plt.figure(figsize=(4, 5))  # narrower figure
    # bottom segment: inferred_count
    plt.bar(0, inferred_count, color=color_inferred, label="Inferred")
    # top segment: not_inferred_count
    plt.bar(0, not_inferred_count, bottom=inferred_count,
            color=color_not_inferred, label="Not Inferred")

    plt.title(f"{pretty_name(field_name)}\nTotal N={total_count}", fontsize=18)
    plt.xticks([0], [""])
    plt.yticks(fontsize=14)
    plt.ylabel("Count", fontsize=16)
    plt.legend(fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{field_name}_stacked_inferred_bar.png")
    plt.savefig(out_path, dpi=600)
    plt.close()

###############################################################################
# 5) Main script
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_csv", required=True, help="path_fields.csv file path")
    parser.add_argument("--cons_csv", required=True, help="cons_fields.csv file path")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df_path = pd.read_csv(args.path_csv)
    df_cons = pd.read_csv(args.cons_csv)

    # Example fields for demonstration
    horizontal_bar_fields = [
        "Alcohol_Consumption",
        "ECOG_Performance_Status",
        "Karnofsky_Performance_Status",
        "Lymph_Node_Status_Presence_Absence",
        "Smoking_History",
        "p16_Status",
        "Sex"
    ]
    vertical_bar_fields = [
        "Alcohol_Consumption",
        "ECOG_Performance_Status",
        "Karnofsky_Performance_Status",
        "Lymph_Node_Status_Presence_Absence",
        "Smoking_History",
        "p16_Status",
        "Sex"
    ]

    # Merge the sets for iteration
    all_fields = set(horizontal_bar_fields + vertical_bar_fields)

    # A helper to get a column from path or cons
    def get_series(field):
        if field in df_path.columns:
            return df_path[field]
        elif field in df_cons.columns:
            return df_cons[field]
        else:
            return None

    # For each field, produce vertical/horizontal bar (excluding Not Inferred)
    # plus the single stacked bar acknowledging Not Inferred
    for fld in all_fields:
        coldata = get_series(fld)
        if coldata is None:
            continue

        # Vertical
        if fld in vertical_bar_fields:
            make_bar_plot_vertical(coldata, fld, args.out_dir)

        # Horizontal
        if fld in horizontal_bar_fields:
            make_bar_plot_horizontal(coldata, fld, args.out_dir)

        # Single stacked bar for Inferred vs Not Inferred
        make_single_stacked_bar_inferred(coldata, fld, args.out_dir)

    print(f"Plots saved to => {args.out_dir}")


if __name__ == "__main__":
    main()


# Usage example
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/analyze_extracted_fields.py \
#     --path_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Path_Structured/path_fields.csv" \
#     --cons_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Cons_Structured/cons_fields.csv" \
#     --out_dir  "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Structured_Fields_plots/Exp14"
