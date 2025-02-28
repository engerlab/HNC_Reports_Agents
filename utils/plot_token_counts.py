#!/usr/bin/env python3
"""
Plot Histograms of Token Counts with a Vertical Threshold

This script reads a CSV containing 'text_tokens' and 'total_needed' columns,
then generates three histograms:

1. hist_text_tokens.png
   - A histogram of the 'text_tokens' column with a red dashed line at 2048.

2. hist_total_needed.png
   - A histogram of the 'total_needed' column with a red dashed line at 2048.

3. hist_overlapped_text_tokens_vs_total_needed.png
   - Overlapping histograms for 'text_tokens' and 'total_needed'
     on the same figure, also with a red dashed line at 2048.

Command-Line Arguments:
  --input_csv   : Path to a CSV file containing 'text_tokens' and 'total_needed'.
  --output_dir  : Folder to store the resulting PNG files.

Example:
  python plot_token_counts.py \
    --input_csv "/path/to/precompute_tokens.csv" \
    --output_dir "/path/to/histograms"

The script checks for the required columns, creates separate PNG outputs, 
and places a vertical threshold line at x=2048 for each plot.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of text_tokens and total_needed from a CSV file, with a vertical line at 2048."
    )
    parser.add_argument(
        "--input_csv", required=True,
        help="Path to the precompute_tokens.csv (containing columns 'text_tokens' and 'total_needed')."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to store the resulting .png histogram files."
    )
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.input_csv)

    # Ensure required columns exist
    for col in ["text_tokens", "total_needed"]:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV: {args.input_csv}")
            return

    # Create output dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Histogram of text_tokens
    plt.figure(figsize=(6,4))
    sns.histplot(df["text_tokens"], bins=30, kde=False, color="steelblue")
    plt.axvline(x=2048, color='black', linestyle='--', linewidth=1, label='2048 default token')
    plt.title("Histogram of text_tokens")
    plt.xlabel("text_tokens")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    outpath1 = os.path.join(args.output_dir, "hist_text_tokens.png")
    plt.savefig(outpath1, dpi=600)
    plt.close()

    # 2) Histogram of total_needed
    plt.figure(figsize=(6,4))
    sns.histplot(df["total_needed"], bins=30, kde=False, color="coral")
    plt.axvline(x=2048, color='black', linestyle='--', linewidth=1, label='2048 default token')
    plt.title("Histogram of total_needed")
    plt.xlabel("total_needed")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    outpath2 = os.path.join(args.output_dir, "hist_total_needed.png")
    plt.savefig(outpath2, dpi=600)
    plt.close()

    # 3) Overlapping histograms of text_tokens vs. total_needed
    plt.figure(figsize=(6,4))
    sns.histplot(df["text_tokens"], bins=30, kde=False, color="steelblue", alpha=0.5, label="text_tokens")
    sns.histplot(df["total_needed"], bins=30, kde=False, color="coral", alpha=0.5, label="total_needed")
    plt.axvline(x=2048, color='black', linestyle='--', linewidth=3, label='2048 default token')
    plt.title("text_tokens vs. total_needed", fontsize=20)
    plt.xlabel("Token Count", fontsize=17)
    plt.ylabel("Report Cases Count", fontsize=17)
    plt.legend()
    plt.tight_layout()
    outpath3 = os.path.join(args.output_dir, "hist_overlapped_text_tokens_vs_total_needed.png")
    plt.savefig(outpath3, dpi=600)
    plt.close()

    print("Plots created:")
    print(" ", outpath1)
    print(" ", outpath2)
    print(" ", outpath3)

if __name__ == "__main__":
    main()

# Usage Example:
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts.py \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"
