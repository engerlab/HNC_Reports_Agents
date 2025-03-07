#!/usr/bin/env python3
"""
analyze_extracted_fields.py (Modified)

Generates:
  - Vertical bar plots
  - Horizontal bar plots
  - Histograms (for numeric fields)
with a color gradient, large fonts, and DPI=600. Renames "nan" => "Not Inferred."

Particular fields of interest (examples):
  1) Horizontal bar plot for:
     - Alcohol_Consumption
     - ECOG_Performance_Status, Karnofsky_Performance_Status
     - Lymph_Node_Status_Presence_Absence
     - Pack_Years
     - Smoking_History
  2) Keep original vertical bar plots, too, for some fields
  3) Histograms for numeric fields (Charlson_Comorbidity_Score, etc.)
  4) Rename "nan" => "Not Inferred"

Usage:
  python analyze_extracted_fields.py \
    --path_csv "/Data/.../path_fields.csv" \
    --cons_csv "/Data/.../cons_fields.csv" \
    --out_dir  "/Data/.../Structured_Fields_plots/Exp14"

Next steps:
  - Draw connection between fields and outcomes
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def get_color_list(n, cmap_name="Blues"):
    """
    Create a list of n distinct colors from the given colormap.
    """
    # For a categorical palette, sample the colormap in n discrete steps.
    cmap = cm.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]

def rename_nan_to_inferred(series):
    """
    Replaces any 'nan' string (case-insensitive) with "Not Inferred"
    in the index (for bar plots) or in the values (for direct Series).
    """
    cleaned = []
    for val in series:
        sval = str(val).strip().lower()
        if sval == "nan":
            cleaned.append("Not Inferred")
        else:
            cleaned.append(val)
    return pd.Series(cleaned, index=series.index)

def make_bar_plot_vertical(series, field_name, out_dir):
    """
    Standard (vertical) bar plot with a color gradient. 
    Also rename 'nan' => 'Not Inferred' in the categories.
    """
    # Value counts (including NaN)
    counts = series.value_counts(dropna=False)
    # Replace "nan" => "Not Inferred"
    cleaned_index = []
    for c in counts.index.astype(str):
        if c.strip().lower() == "nan":
            cleaned_index.append("Not Inferred")
        else:
            cleaned_index.append(c)
    counts.index = cleaned_index

    # Sort categories by frequency descending (optional)
    # counts = counts.sort_values(ascending=False)

    # Prepare colors
    color_list = get_color_list(len(counts))

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=color_list)
    plt.title(f"{field_name}", fontsize=20)
    plt.xlabel(field_name, fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{field_name}_bar_vertical.png")
    plt.savefig(out_path, dpi=600)
    plt.close()

def make_bar_plot_horizontal(series, field_name, out_dir):
    """
    Horizontal bar plot with color gradient, rename 'nan' => 'Not Inferred'.
    The largest category will appear at the top if we sort ascending.
    """
    counts = series.value_counts(dropna=False)
    # rename "nan" => "Not Inferred"
    cleaned_index = []
    for c in counts.index.astype(str):
        if c.strip().lower() == "nan":
            cleaned_index.append("Not Inferred")
        else:
            cleaned_index.append(c)
    counts.index = cleaned_index

    # sort ascending so largest bar is on top
    counts = counts.sort_values(ascending=True)

    color_list = get_color_list(len(counts))

    plt.figure(figsize=(8, 5))
    plt.barh(counts.index, counts.values, color=color_list)
    plt.title(f"{field_name}", fontsize=20)
    plt.xlabel("Count", fontsize=18)
    plt.ylabel(field_name, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{field_name}_bar_horizontal.png")
    plt.savefig(out_path, dpi=600)
    plt.close()

def make_hist_plot(series, field_name, out_dir, bins=10):
    """
    Histogram for numeric distribution. 
    Force integer ticks if the range is small.
    """
    numeric_vals = pd.to_numeric(series, errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(numeric_vals, bins=bins, color="skyblue", edgecolor="black")
    plt.title(f"{field_name} Histogram", fontsize=20)
    plt.xlabel(field_name, fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # If the data are small integer range, force integer ticks
    if len(numeric_vals) > 0:
        min_val, max_val = int(numeric_vals.min()), int(numeric_vals.max())
        # If the range is not huge, set integer ticks
        if max_val - min_val < 15:
            plt.xticks(range(min_val, max_val + 1))

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{field_name}_hist.png")
    plt.savefig(out_path, dpi=600)
    plt.close()

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

    # Example fields
    # 1) Horizontal bar for: Alcohol_Consumption, ECOG_Performance_Status, Karnofsky_Performance_Status,
    #    Lymph_Node_Status_Presence_Absence, Pack_Years, Smoking_History
    # 2) Keep original vertical bar plots, too, for certain fields
    # 3) Hist for numeric fields

    # Let's define the fields:
    horizontal_bar_fields = [
        "Alcohol_Consumption",
        "ECOG_Performance_Status",
        "Karnofsky_Performance_Status",
        "Lymph_Node_Status_Presence_Absence",
        "Pack_Years",
        "Smoking_History",
    ]
    # We'll also do a vertical bar for these same fields, to "keep the original"
    vertical_bar_fields = horizontal_bar_fields[:]  # copy

    # Some fields are numeric => we also do a histogram (like Pack_Years, ECOG, etc.)
    numeric_hist_fields = [
        "Charlson_Comorbidity_Score",
        "ECOG_Performance_Status",
        "Karnofsky_Performance_Status",
        "Pack_Years",
    ]

    # Additional small fields that have 2-3 categories: p16_Status, Sex, etc.
    # We can do a bar plot for those as well:
    small_cat_fields = ["p16_Status", "Sex"]
    # We'll do vertical bar for them, plus horizontal if you like
    # For demonstration, let's do both.
    horizontal_bar_fields.extend(small_cat_fields)
    vertical_bar_fields.extend(small_cat_fields)

    # Now produce plots for each field if it exists in either df_path or df_cons
    # We'll unify them in the sense that if the field is in df_path, we plot from that, else df_cons.

    def plot_field(df, field):
        if field not in df.columns:
            return

        # 1) If field is in vertical_bar_fields => do vertical bar
        if field in vertical_bar_fields:
            make_bar_plot_vertical(df[field], field, args.out_dir)

        # 2) If field is in horizontal_bar_fields => do horizontal bar
        if field in horizontal_bar_fields:
            make_bar_plot_horizontal(df[field], field, args.out_dir)

        # 3) If field is in numeric_hist_fields => do a histogram
        if field in numeric_hist_fields:
            make_hist_plot(df[field], field, args.out_dir, bins=10)

    # We'll gather all columns across path/cons in a single list, then handle duplicates
    all_fields = set(df_path.columns) | set(df_cons.columns)

    for field in sorted(all_fields):
        # Decide which dataframe to use (some fields might appear in both)
        if field in df_path.columns:
            plot_field(df_path, field)
        elif field in df_cons.columns:
            plot_field(df_cons, field)

    print(f"Plots saved to: {args.out_dir}")

if __name__ == "__main__":
    main()


# Usage example
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/analyze_extracted_fields.py \
#     --path_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Path_Structured/path_fields.csv" \
#     --cons_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Cons_Structured/cons_fields.csv" \
#     --out_dir  "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Structured_Fields_plots/Exp14"
