#!/usr/bin/env python3
"""
Plot Token Counts for HNC Summaries (4 Experiment Modes)

Usage Example:
  python plot_token_counts.py \
    --exp_mode 2 \
    --input_csv "/Data/Token_Counts/precompute_tokens_mode2.csv" \
    --output_dir "/Data/Token_Counts/plots"

The script inspects the CSV columns according to the chosen exp_mode:
 Mode 1 => 'text_tokens', 'total_needed'
 Mode 2 => 'extr_text_tokens','extr_total_needed','cot_text_tokens','cot_total_needed'
 Mode 3 => typically has 'text_tokens','total_needed','report_type' or 'subcall'
 Mode 4 => typically has path_extr_..., path_cot_..., cons_extr_..., cons_cot_...

We produce a subfolder inside --output_dir named "mode{X}_plots" with the relevant .png files.
A dashed vertical line at x=2048 is drawn on each histogram to compare with the default ChatOllama limit.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser("Plot token counts for different experiment modes.")
    parser.add_argument("--exp_mode", type=int, required=True, choices=[1,2,3,4],
                        help="Which experiment mode (1..4) to handle.")
    parser.add_argument("--input_csv", required=True,
                        help="Path to the precompute_tokens_modeX.csv file.")
    parser.add_argument("--output_dir", required=True,
                        help="Base directory where 'modeX_plots' subfolder is created.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)

    # Create subfolder: modeX_plots
    mode_str = f"mode{args.exp_mode}_plots"
    outdir = os.path.join(args.output_dir, mode_str)
    os.makedirs(outdir, exist_ok=True)

    # Decide how to handle each mode
    if args.exp_mode == 1:
        # Expect columns: text_tokens, total_needed
        # We'll make 3 plots:
        # 1) hist_text_tokens
        # 2) hist_total_needed
        # 3) overlap of text_tokens vs total_needed
        required = ["text_tokens","total_needed"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Mode 1 missing columns: {missing} in {args.input_csv}.")
            return
        plot_hist_1col(df["text_tokens"], outdir, "text_tokens", "Mode 1: text_tokens")
        plot_hist_1col(df["total_needed"], outdir, "total_needed", "Mode 1: total_needed")
        # overlap
        plot_hist_overlap(df["text_tokens"], df["total_needed"], outdir,
                          labelA="text_tokens", labelB="total_needed",
                          title="Mode 1")

    elif args.exp_mode == 2:
        # 2-step combined => expect extr_text_tokens,extr_total_needed,cot_text_tokens,cot_total_needed
        needed = ["extr_text_tokens","extr_total_needed","cot_text_tokens","cot_total_needed"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"Mode 2 missing columns: {missing} in {args.input_csv}.")
            return
        # produce separate hist for extr_text_tokens, extr_total_needed
        plot_hist_1col(df["extr_text_tokens"], outdir, "extr_text_tokens", "Mode 2: extr_text_tokens")
        plot_hist_1col(df["extr_total_needed"], outdir, "extr_total_needed", "Mode 2: extr_total_needed")
        # produce separate hist for cot_text_tokens, cot_total_needed
        plot_hist_1col(df["cot_text_tokens"], outdir, "cot_text_tokens", "Mode 2: cot_text_tokens")
        plot_hist_1col(df["cot_total_needed"], outdir, "cot_total_needed", "Mode 2: cot_total_needed")

        # overlapping extr_total_needed vs. cot_total_needed
        plot_hist_overlap(df["extr_total_needed"], df["cot_total_needed"], outdir,
                          labelA="extr_total_needed", labelB="cot_total_needed",
                          title="Mode 2")

    elif args.exp_mode == 3:
        # separate single-step => we often have "text_tokens","total_needed" plus a "subcall" or "report_type"
        # Possibly do separate hist by subcall=path vs subcall=cons
        # We'll do a subset approach if found.
        needed = ["text_tokens","total_needed"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"Mode 3 missing columns: {missing} in {args.input_csv}.")
            return

        # We can do an overall hist of text_tokens, total_needed
        # Then possibly do path-only, consult-only
        plot_hist_1col(df["text_tokens"], outdir, "text_tokens_all", "Mode 3: text_tokens (All)")

        plot_hist_1col(df["total_needed"], outdir, "total_needed_all", "Mode 3: total_needed (All)")

        # overlap
        plot_hist_overlap(df["text_tokens"], df["total_needed"], outdir,
                          labelA="text_tokens", labelB="total_needed",
                          title="Mode 3")

        # If we have a 'subcall' or 'report_type' column, let's do separate hist
        if "subcall" in df.columns:
            # path rows
            path_df = df[df["subcall"].str.contains("path", case=False, na=False)]
            if not path_df.empty:
                plot_hist_1col(path_df["text_tokens"], outdir, "text_tokens_path", "Mode3: Path text_tokens")
                plot_hist_1col(path_df["total_needed"], outdir, "total_needed_path", "Mode3: Path total_needed")
                plot_hist_overlap(path_df["text_tokens"], path_df["total_needed"], outdir,
                                  labelA="path_text_tokens", labelB="path_total_needed",
                                  title="Mode3")

            # consult rows
            cons_df = df[df["subcall"].str.contains("cons", case=False, na=False)]
            if not cons_df.empty:
                plot_hist_1col(cons_df["text_tokens"], outdir, "text_tokens_cons", "Mode3: Consult text_tokens")
                plot_hist_1col(cons_df["total_needed"], outdir, "total_needed_cons", "Mode3: Consult total_needed")
                plot_hist_overlap(cons_df["text_tokens"], cons_df["total_needed"], outdir,
                                  labelA="cons_text_tokens", labelB="cons_total_needed",
                                  title="Mode3")

    else:
        # mode 4 => separate two-step => path_extr_text_tokens, path_extr_total_needed, etc.
        # We'll produce hist for each, plus optional overlap combos
        # We'll try to produce 8 single-col hist: path_extr_text_tokens, path_extr_total_needed,
        # path_cot_text_tokens, path_cot_total_needed, cons_extr_text_tokens, cons_extr_total_needed,
        # cons_cot_text_tokens, cons_cot_total_needed
        # Then do overlap of e.g. path_extr_total_needed vs. path_cot_total_needed, cons_extr_total_needed vs. cons_cot_total_needed
        neededCols = [
            "path_extr_text_tokens","path_extr_total_needed",
            "path_cot_text_tokens","path_cot_total_needed",
            "cons_extr_text_tokens","cons_extr_total_needed",
            "cons_cot_text_tokens","cons_cot_total_needed"
        ]
        missing = [c for c in neededCols if c not in df.columns]
        if missing:
            print(f"Mode 4 missing columns: {missing} in {args.input_csv}.")
            return

        # 8 single-col hist
        plot_hist_1col(df["path_extr_text_tokens"], outdir, "path_extr_text_tokens", "Mode4: path_extr_text_tokens")
        plot_hist_1col(df["path_extr_total_needed"], outdir, "path_extr_total_needed", "Mode4: path_extr_total_needed")
        plot_hist_1col(df["path_cot_text_tokens"], outdir, "path_cot_text_tokens", "Mode4: path_cot_text_tokens")
        plot_hist_1col(df["path_cot_total_needed"], outdir, "path_cot_total_needed", "Mode4: path_cot_total_needed")

        plot_hist_1col(df["cons_extr_text_tokens"], outdir, "cons_extr_text_tokens", "Mode4: cons_extr_text_tokens")
        plot_hist_1col(df["cons_extr_total_needed"], outdir, "cons_extr_total_needed", "Mode4: cons_extr_total_needed")
        plot_hist_1col(df["cons_cot_text_tokens"], outdir, "cons_cot_text_tokens", "Mode4: cons_cot_text_tokens")
        plot_hist_1col(df["cons_cot_total_needed"], outdir, "cons_cot_total_needed", "Mode4: cons_cot_total_needed")

        # Overlaps: path_extr_total_needed vs path_cot_total_needed
        plot_hist_overlap(df["path_extr_total_needed"], df["path_cot_total_needed"], outdir,
                          labelA="path_extr_total_needed", labelB="path_cot_total_needed",
                          title="Mode 4")

        # Overlaps: cons_extr_total_needed vs cons_cot_total_needed
        plot_hist_overlap(df["cons_extr_total_needed"], df["cons_cot_total_needed"], outdir,
                          labelA="cons_extr_total_needed", labelB="cons_cot_total_needed",
                          title="Mode4")


    print(f"Plots saved in => {outdir}")

def plot_hist_1col(series, outdir, name_prefix, title):
    """
    Create a single histogram for 'series', with a vertical line at 2048.
    Save as name_prefix + '.png' in outdir.
    """
    plt.figure(figsize=(6,4))
    sns.histplot(series, bins=30, kde=False, color="steelblue", alpha=0.7)
    plt.axvline(x=2048, color='black', linestyle='--', linewidth=3, label='2048 tokens')
    plt.title(title, fontsize = 20)
    plt.xlabel(name_prefix, fontsize = 17)
    plt.ylabel("Cases Count", fontsize=17)
    plt.legend(fontsize = 14)
    plt.tight_layout()
    outpath = os.path.join(outdir, f"{name_prefix}.png")
    plt.savefig(outpath, dpi=600)
    plt.close()

def plot_hist_overlap(seriesA, seriesB, outdir, labelA="A", labelB="B", title="Overlap"):
    """
    Overlapping histogram for two series.
    """
    plt.figure(figsize=(6,4))
    sns.histplot(seriesA, bins=30, kde=False, color="steelblue", alpha=0.5, label=labelA)
    sns.histplot(seriesB, bins=30, kde=False, color="coral", alpha=0.5, label=labelB)
    plt.axvline(x=2048, color='black', linestyle='--', linewidth=3, label='2048 tokens')
    plt.title(title, fontsize = 20)
    plt.xlabel("Token Count", fontsize = 17)
    plt.ylabel("Cases Count", fontsize = 17)
    plt.legend(fontsize = 14)
    plt.tight_layout()
    outpath = os.path.join(outdir, f"overlap_{labelA}_vs_{labelB}.png")
    plt.savefig(outpath, dpi=600)
    plt.close()

if __name__ == "__main__":
    main()


# Usage Example:

# Mode 1 => Combined text + Single-step prompt
# Mode 2 => Combined text + Two-step prompt (extraction + CoT)
# Mode 3 => Separate texts (Pathology + Consultation) + Single-step prompt
# Mode 4 => Separate texts (Pathology + Consultation) + Two-step prompt

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts.py \
#   --exp_mode 1 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode1.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"


# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts.py \
#   --exp_mode 2 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode2.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts.py \
#   --exp_mode 3 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode3.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts.py \
#   --exp_mode 4 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode4.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"
