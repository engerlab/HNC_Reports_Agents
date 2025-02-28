#!/usr/bin/env python3
"""
Plot Token Counts for HNC Summaries (4 Experiment Modes)

This script preserves the 'individual histograms' for each column but adjusts
the "overlap" plotting logic:
  Mode 1 => Combined text + Single-step prompt
  Mode 2 => Combined text + Two-step prompt (extraction + CoT)
  Mode 3 => Separate texts (Pathology + Consultation) + Single-step prompt
  Mode 4 => Separate texts (Pathology + Consultation) + Two-step prompt

Mode 1:
 - Hist: text_tokens, total_needed
 - Overlap: (skipped) => "only total_needed is compared to 2048"

Mode 2:
 - Hist: extr_text_tokens, extr_total_needed, cot_text_tokens, cot_total_needed
 - Overlap: extr_total_needed vs. cot_total_needed

Mode 3:
 - Hist (path): path_text_tokens, path_total_needed
 - Hist (cons): cons_text_tokens, cons_total_needed
 - Overlap: path_total_needed vs. cons_total_needed

Mode 4:
 - Hist (8 columns):
   path_extr_text_tokens, path_extr_total_needed,
   path_cot_text_tokens, path_cot_total_needed,
   cons_extr_text_tokens, cons_extr_total_needed,
   cons_cot_text_tokens, cons_cot_total_needed
 - Overlap: a single plot with 4 lines:
    1) path_extr_total_needed
    2) cons_extr_total_needed
    3) path_cot_total_needed
    4) cons_cot_total_needed
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(
        description="Plot token counts for different experiment modes, subcall-based for Mode 4."
    )
    parser.add_argument("--exp_mode", type=int, required=True, choices=[1,2,3,4],
                        help="Which experiment mode (1..4) to handle.")
    parser.add_argument("--input_csv", required=True,
                        help="Path to the precompute_tokens_modeX.csv file.")
    parser.add_argument("--output_dir", required=True,
                        help="Base directory where 'modeX_plots' subfolder is created.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    mode_str = f"mode{args.exp_mode}_plots"
    outdir = os.path.join(args.output_dir, mode_str)
    os.makedirs(outdir, exist_ok=True)

    if args.exp_mode == 1:
        # Need: text_tokens, total_needed
        needed_cols = ["text_tokens","total_needed"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            print(f"[Mode 1] Missing columns: {missing}")
            return

        # Indiv hist
        plot_hist_1col(df["text_tokens"], outdir, "text_tokens", "Mode 1: text_tokens")
        plot_hist_1col(df["total_needed"], outdir, "total_needed", "Mode 1: Combined text + Single-step prompt")

        # Overlap => skip
        print("[Mode 1] Overlap: Skipped, only total_needed is relevant vs. 2048")

    elif args.exp_mode == 2:
        # columns: extr_text_tokens, extr_total_needed, cot_text_tokens, cot_total_needed
        needed = ["extr_text_tokens","extr_total_needed","cot_text_tokens","cot_total_needed"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[Mode 2] Missing columns: {missing}")
            return

        # individual
        plot_hist_1col(df["extr_text_tokens"], outdir, "extr_text_tokens", "Mode 2: extr_text_tokens")
        plot_hist_1col(df["extr_total_needed"], outdir, "extr_total_needed", "Mode 2: extr_total_needed")
        plot_hist_1col(df["cot_text_tokens"], outdir, "cot_text_tokens", "Mode 2: cot_text_tokens")
        plot_hist_1col(df["cot_total_needed"], outdir, "cot_total_needed", "Mode 2: cot_total_needed")

        # overlap => extr_total_needed vs. cot_total_needed
        plot_hist_overlap(
            df["extr_total_needed"], df["cot_total_needed"], outdir,
            labelA="extr_total_needed", labelB="cot_total_needed",
            title="Mode 2: Combined text + Two-step prompt"
        )

    elif args.exp_mode == 3:
        # columns: subcall, text_tokens, total_needed
        needed = ["subcall","text_tokens","total_needed"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[Mode 3] Missing columns: {missing}")
            return

        # separate path vs cons
        path_df = df[df["subcall"].str.contains("path", case=False, na=False)]
        cons_df = df[df["subcall"].str.contains("cons", case=False, na=False)]

        if not path_df.empty:
            plot_hist_1col(path_df["text_tokens"], outdir, "path_text_tokens", "Mode3: path_text_tokens")
            plot_hist_1col(path_df["total_needed"], outdir, "path_total_needed", "Mode3: path_total_needed")
        if not cons_df.empty:
            plot_hist_1col(cons_df["text_tokens"], outdir, "cons_text_tokens", "Mode3: consult_text_tokens")
            plot_hist_1col(cons_df["total_needed"], outdir, "cons_total_needed", "Mode3: consult_total_needed")

        # overlap => path_total_needed vs cons_total_needed
        if not path_df.empty and not cons_df.empty:
            plot_hist_overlap(
                path_df["total_needed"], cons_df["total_needed"], outdir,
                labelA="path_total_needed", labelB="cons_total_needed",
                title="Mode 3: Separate texts + Single-step prompt"
            )

    else:
        # mode 4 => subcall in {path_extraction, path_cot, cons_extraction, cons_cot}
        # each row has either extr_... or cot_... columns
        needed = ["subcall","num_input_characters","extr_text_tokens","extr_total_needed","cot_text_tokens","cot_total_needed"]
        for c in needed:
            if c not in df.columns:
                print(f"[Mode 4] Missing column {c} in {args.input_csv}.")
        # We'll produce hist for each subcall if present

        # subcalls
        path_extr_df = df[df["subcall"] == "path_extraction"]
        path_cot_df  = df[df["subcall"] == "path_cot"]
        cons_extr_df = df[df["subcall"] == "cons_extraction"]
        cons_cot_df  = df[df["subcall"] == "cons_cot"]

        # 1) path_extraction => extr_text_tokens, extr_total_needed
        if not path_extr_df.empty:
            plot_hist_1col(path_extr_df["extr_text_tokens"], outdir, "path_extr_text_tokens", "Mode4: path_extraction text_tokens")
            plot_hist_1col(path_extr_df["extr_total_needed"], outdir, "path_extr_total_needed", "Mode4: path_extraction total_needed")

        # 2) path_cot => cot_text_tokens, cot_total_needed
        if not path_cot_df.empty:
            plot_hist_1col(path_cot_df["cot_text_tokens"], outdir, "path_cot_text_tokens", "Mode4: path_cot text_tokens")
            plot_hist_1col(path_cot_df["cot_total_needed"], outdir, "path_cot_total_needed", "Mode4: path_cot total_needed")

        # 3) cons_extraction => extr_text_tokens, extr_total_needed
        if not cons_extr_df.empty:
            plot_hist_1col(cons_extr_df["extr_text_tokens"], outdir, "cons_extr_text_tokens", "Mode4: cons_extraction text_tokens")
            plot_hist_1col(cons_extr_df["extr_total_needed"], outdir, "cons_extr_total_needed", "Mode4: cons_extraction total_needed")

        # 4) cons_cot => cot_text_tokens, cot_total_needed
        if not cons_cot_df.empty:
            plot_hist_1col(cons_cot_df["cot_text_tokens"], outdir, "cons_cot_text_tokens", "Mode4: cons_cot text_tokens")
            plot_hist_1col(cons_cot_df["cot_total_needed"], outdir, "cons_cot_total_needed", "Mode4: cons_cot total_needed")

        # Overlap => single figure with 4 distributions:
        #   path_extr_df["extr_total_needed"]
        #   path_cot_df ["cot_total_needed"]
        #   cons_extr_df["extr_total_needed"]
        #   cons_cot_df ["cot_total_needed"]
        plt.figure(figsize=(6,4))
        if not path_extr_df.empty:
            sns.histplot(path_extr_df["extr_total_needed"].dropna(), bins=30, kde=False, color="steelblue",
                         alpha=0.4, label="path_extraction")
        if not path_cot_df.empty:
            sns.histplot(path_cot_df["cot_total_needed"].dropna(), bins=30, kde=False, color="red",
                         alpha=0.4, label="path_cot")
        if not cons_extr_df.empty:
            sns.histplot(cons_extr_df["extr_total_needed"].dropna(), bins=30, kde=False, color="green",
                         alpha=0.4, label="cons_extraction")
        if not cons_cot_df.empty:
            sns.histplot(cons_cot_df["cot_total_needed"].dropna(), bins=30, kde=False, color="purple",
                         alpha=0.4, label="cons_cot")

        plt.axvline(x=2048, color='black', linestyle='--', linewidth=2, label='2048 tokens')
        plt.title("Mode4: Separate texts + Two-step prompt",fontsize=14)
        plt.xlabel("Token Count", fontsize=17)
        plt.ylabel("Count",fontsize=17)
        plt.legend(fontsize=14)
        plt.tight_layout()
        overlap_out = os.path.join(outdir, "overlap_mode4_subcalls_total_needed.png")
        plt.savefig(overlap_out, dpi=600)
        plt.close()

    print(f"[Mode {args.exp_mode}] Plots saved to => {outdir}")


def plot_hist_1col(series, outdir, name_prefix, title):
    """
    Single histogram with vertical line at 2048.
    """
    ser = series.dropna()
    if ser.empty:
        print(f"No data for {name_prefix}, skipping plot.")
        return

    plt.figure(figsize=(6,4))
    sns.histplot(ser, bins=30, kde=False, color="steelblue", alpha=0.7)
    plt.axvline(x=2048, color='black', linestyle='--', linewidth=2, label='2048 tokens')
    plt.title(title,fontsize=14)
    plt.xlabel(name_prefix,fontsize=17)
    plt.ylabel("Count",fontsize=17)
    plt.legend(fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(outdir, f"{name_prefix}.png")
    plt.savefig(outpath, dpi=600)
    plt.close()

def plot_hist_overlap(seriesA, seriesB, outdir, labelA="A", labelB="B", title="Overlap"):
    """
    Overlap histogram for two series
    """
    a_drop = seriesA.dropna()
    b_drop = seriesB.dropna()
    if a_drop.empty and b_drop.empty:
        print(f"No data for overlap {labelA} vs {labelB}, skipping.")
        return

    plt.figure(figsize=(6,4))
    if not a_drop.empty:
        sns.histplot(a_drop, bins=30, kde=False, color="steelblue", alpha=0.5, label=labelA)
    if not b_drop.empty:
        sns.histplot(b_drop, bins=30, kde=False, color="coral", alpha=0.5, label=labelB)

    plt.axvline(x=2048, color='black', linestyle='--', linewidth=2, label='2048 tokens')
    plt.title(title,fontsize=14)
    plt.xlabel("Token Count", fontsize=17)
    plt.ylabel("Count",fontsize=17)
    plt.legend(fontsize=14)
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

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts2.py \
#   --exp_mode 1 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode1.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"


# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts2.py \
#   --exp_mode 2 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode2.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts2.py \
#   --exp_mode 3 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode3.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/plot_token_counts2.py \
#   --exp_mode 4 \
#   --input_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode4.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/histograms"
