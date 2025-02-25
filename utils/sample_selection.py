#!/usr/bin/env python3
"""
sample_selection.py

Adds:
  1) 'category' column in final CSV: path_only, consult_only, path_cons.
  2) --n_path, --n_consult, --n_both arguments to specify how many from each category.
  3) --both_cutoff, --path_cutoff, --cons_cutoff to control numeric thresholds in cutoff mode
     (replacing the fixed 10, 15, 15 from the old script).

Usage Examples:
  1) Stratified mode, still picks 5/18/27:
     python sample_selection.py \
       --base_dir "/Data/.../Exp13" \
       --path_only "/Data/.../path_only_ids.csv" \
       --consult_only "/Data/.../consult_only_ids.csv" \
       --both "/Data/.../both_ids.csv" \
       --output_dir "/Data/.../CaseSelection" \
       --mode stratified \
       --n_path 5 --n_consult 18 --n_both 27

  2) Cutoff mode, user sets both_cutoff=10, path_cutoff=15, cons_cutoff=15:
     python sample_selection.py \
       --base_dir "/Data/.../Exp13" \
       --path_only "/Data/.../path_only_ids.csv" \
       --consult_only "/Data/.../consult_only_ids.csv" \
       --both "/Data/.../both_ids.csv" \
       --output_dir "/Data/.../CaseSelection" \
       --mode cutoff \
       --n_path 5 --n_consult 18 --n_both 27 \
       --both_cutoff 10 --path_cutoff 15 --cons_cutoff 15
"""

import os
import re
import argparse
import logging
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_ids(csv_path):
    df = pd.read_csv(csv_path)
    return set(map(str, df["patient_id"].tolist()))

def count_not_inferred(summary_file):
    if not os.path.isfile(summary_file):
        return None
    c = 0
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            if "not inferred" in line.strip().lower(): #converts to lowercase and removes leading/trailing whitespaces before seeing words
                c += 1
    return c

def create_scatter_plot(df, category_name, output_dir):
    df2 = df.dropna(subset=["num_input_tokens","not_inferred_count"])
    plt.figure(figsize=(6,4))
    plt.scatter(df2["num_input_tokens"], df2["not_inferred_count"], alpha=0.6)
    plt.xlabel("num_input_tokens")
    plt.ylabel("not_inferred_count")
    plt.title(f"{category_name} scatter")
    out_path = os.path.join(output_dir, f"{category_name}_scatter.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Scatter plot saved: {out_path}")

def create_jointplot(df, category_name, output_dir):
    """
    Seaborn jointplot with color-coded bins for not_inferred_count:
      - low: <10
      - medium: 10-20
      - high: >20
    (These color bins are just for illustration, not necessarily the same 
     as the logic used in the 'cutoff' or 'stratified' approach.)
    """
    df2 = df.dropna(subset=["num_input_tokens","not_inferred_count"]).copy()

    def categorize_not_inferred(x):
        if x < 10:
            return "low"
        elif x <= 20:
            return "medium"
        else:
            return "high"

    df2["bin_inferred"] = df2["not_inferred_count"].apply(categorize_not_inferred)

    g = sns.jointplot(
        data=df2,
        x="num_input_tokens",
        y="not_inferred_count",
        hue="bin_inferred",
        alpha=0.6
    )
    g.fig.suptitle(f"{category_name} JointPlot by not_inferred bins", y=1.02)
    out_path = os.path.join(output_dir, f"{category_name}_joint.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Jointplot saved: {out_path}")

def build_category_df(cat_name, cat_ids, df_times, summaries_dir):
    """
    For each ID in cat_ids => 
      columns: [patient_id, category, process_time_ms, num_input_characters, num_input_tokens, not_inferred_count]
    'category' is one of "path_only", "consult_only", "path_cons".
    """
    subset = df_times[df_times["patient_id"].isin(cat_ids)].copy()
    not_inferred_counts = []
    for idx, row in subset.iterrows():
        pid = str(row["patient_id"])
        summary_file = os.path.join(summaries_dir, pid, "path_consult_reports_summary.txt")
        c = count_not_inferred(summary_file)
        not_inferred_counts.append(c if c is not None else np.nan)
    subset["not_inferred_count"] = not_inferred_counts
    # Add a 'category' column
    subset["category"] = cat_name
    subset = subset[[
        "patient_id","category","process_time_ms","num_input_characters","num_input_tokens","not_inferred_count"
    ]]
    subset.sort_values(by="not_inferred_count", inplace=True, na_position="last")
    return subset

def stratify_by_not_inferred(df, sample_size, category_label):
    """
    Bins => 0-10 => low, 11-20 => med, 21-30 => high
    We pick about sample_size/3 from each bin if possible.
    """
    bins = [(0,10), (11,20), (21,30)]
    out_frames = []
    leftover = sample_size
    for bin_range in bins:
        low, high = bin_range
        sub = df[(df["not_inferred_count"]>=low) & (df["not_inferred_count"]<=high)].copy()
        if len(sub)==0:
            continue
        portion = sample_size // 3
        pick = min(portion, len(sub))
        sample_df = sub.sample(pick, random_state=42)
        leftover -= pick
        out_frames.append(sample_df)

    if leftover>0:
        # fill from the entire set not chosen yet
        chosen_ids = set()
        for tmp in out_frames:
            chosen_ids.update(tmp["patient_id"].tolist())
        sub_remaining = df[~df["patient_id"].isin(chosen_ids)]
        if len(sub_remaining)>0:
            leftover_pick = min(leftover, len(sub_remaining))
            out_frames.append(sub_remaining.sample(leftover_pick, random_state=123))

    final_df = pd.concat(out_frames, ignore_index=True).drop_duplicates(subset=["patient_id"])
    final_df.sort_values(by="not_inferred_count", inplace=True)
    logger.info(f"[stratify] {category_label}: requested {sample_size}, got {len(final_df)}")
    return final_df

def cutoff_selection(df, sample_size, category_type, path_cutoff, cons_cutoff, both_cutoff):
    """
    In cutoff mode, user sets a numeric threshold for each category:
       path_cons  => both_cutoff
       path_only  => path_cutoff
       consult_only => cons_cutoff
    Then we sub-bin within that allowable range.
    """
    if category_type == "path_cons":
        # "both"
        df_filt = df[df["not_inferred_count"] < both_cutoff]
        # We'll define 2 sub-bins from 0..(both_cutoff-1), 
        # e.g. if both_cutoff=10 => bins (0,5),(6,9)
        # if both_cutoff=12 => (0,5),(6,11), etc.
        sub_bins = define_subbins(0, both_cutoff-1, n_subbins=2)
    elif category_type == "path_only":
        df_filt = df[df["not_inferred_count"] < path_cutoff]
        sub_bins = define_subbins(0, path_cutoff-1, n_subbins=3)
    else: # consult_only
        df_filt = df[df["not_inferred_count"] < cons_cutoff]
        sub_bins = define_subbins(0, cons_cutoff-1, n_subbins=3)

    out_frames = []
    leftover = sample_size

    for (low, high) in sub_bins:
        sub = df_filt[(df_filt["not_inferred_count"]>=low) & (df_filt["not_inferred_count"]<=high)].copy()
        if len(sub)==0:
            continue
        # We'll pick sample_size / number_of_subbins
        portion = sample_size // len(sub_bins)
        pick = min(portion, len(sub))
        sample_df = sub.sample(pick, random_state=42)
        leftover -= pick
        out_frames.append(sample_df)

    if leftover>0:
        chosen_ids = set()
        for tmp in out_frames:
            chosen_ids.update(tmp["patient_id"].tolist())
        sub_remaining = df_filt[~df_filt["patient_id"].isin(chosen_ids)]
        if len(sub_remaining)>0:
            leftover_pick = min(leftover, len(sub_remaining))
            out_frames.append(sub_remaining.sample(leftover_pick, random_state=123))

    final_df = pd.concat(out_frames, ignore_index=True).drop_duplicates(subset=["patient_id"])
    final_df.sort_values(by="not_inferred_count", inplace=True)
    logger.info(f"[cutoff+sub-bins] {category_type}: requested {sample_size}, got {len(final_df)} out of {len(df_filt)} possible")
    return final_df

def define_subbins(start_val, end_val, n_subbins=2):
    """
    Helper to define sub-bins from start_val..end_val inclusive.
    Example: if start=0, end=9, n_subbins=2 => [(0,4), (5,9)]
    If n_subbins=3 => [(0,3), (4,6), (7,9)] etc.
    """
    if start_val> end_val:
        return []
    # how wide each sub-bin is:
    total_range = end_val - start_val + 1
    # e.g. for total_range=10, sub-bins=2 => each is about 5 wide
    # for sub-bins=3 => each is about 3 or 4 wide
    step = total_range // n_subbins
    bins = []
    current_low = start_val
    for i in range(n_subbins):
        # for the last bin, we go up to end_val
        if i == n_subbins-1:
            current_high = end_val
        else:
            current_high = current_low + step - 1
        # in case step=0 or negative, guard
        if current_high < current_low:
            break
        bins.append((current_low, current_high))
        current_low = current_high+1
    # e.g. if total_range=10, n_subbins=2 => step=5 => bins => (0,4),(5,9)
    return bins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Folder with processing_times.csv, text_summaries/path_consult_reports")
    parser.add_argument("--path_only", required=True)
    parser.add_argument("--consult_only", required=True)
    parser.add_argument("--both", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["stratified","cutoff"], default="stratified",
                        help="Which selection approach to use.")
    parser.add_argument("--n_path", type=int, default=5, help="How many path_only to pick")
    parser.add_argument("--n_consult", type=int, default=18, help="How many consult_only to pick")
    parser.add_argument("--n_both", type=int, default=27, help="How many path_cons (both) to pick")

    # these replace the old fixed (10, 15, 15)
    parser.add_argument("--both_cutoff", type=int, default=10, help="Cutoff for path_cons category if mode=cutoff")
    parser.add_argument("--path_cutoff", type=int, default=15, help="Cutoff for path_only if mode=cutoff")
    parser.add_argument("--cons_cutoff", type=int, default=15, help="Cutoff for consult_only if mode=cutoff")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the ID sets
    path_ids = load_ids(args.path_only)
    cons_ids = load_ids(args.consult_only)
    both_ids = load_ids(args.both)

    # 2) Load processing_times
    df_times = pd.read_csv(os.path.join(args.base_dir, "processing_times.csv"))
    df_times = df_times[df_times["report_type"]=="path_consult_reports"].copy()
    df_times["patient_id"] = df_times["file"].apply(lambda x: os.path.splitext(x)[0])
    summaries_dir = os.path.join(args.base_dir, "text_summaries", "path_consult_reports")

    # Build each category DataFrame
    df_path_only = build_category_df("path_only", path_ids, df_times, summaries_dir)
    df_cons_only = build_category_df("consult_only", cons_ids, df_times, summaries_dir)
    df_both      = build_category_df("path_cons",   both_ids, df_times, summaries_dir)

    # Save them
    df_path_only.to_csv(os.path.join(args.output_dir,"path_only_stats.csv"), index=False)
    df_cons_only.to_csv(os.path.join(args.output_dir,"consult_only_stats.csv"), index=False)
    df_both     .to_csv(os.path.join(args.output_dir,"both_stats.csv"), index=False)

    # Make scatter + joint plots
    def make_plots(df, cat_name):
        create_scatter_plot(df, cat_name, args.output_dir)
        create_jointplot(df, cat_name, args.output_dir)

    make_plots(df_path_only, "path_only")
    make_plots(df_cons_only, "consult_only")
    make_plots(df_both,      "path_cons")

    # 3) Final selection
    total_picks = args.n_path + args.n_consult + args.n_both

    if args.mode=="stratified":
        logger.info("[MODE] Stratified => bins (0-10), (11-20), (21-30)")
        sel_path = stratify_by_not_inferred(df_path_only,  args.n_path,    "path_only")
        sel_cons = stratify_by_not_inferred(df_cons_only, args.n_consult, "consult_only")
        sel_both = stratify_by_not_inferred(df_both,      args.n_both,    "path_cons")
        outname = f"final_{total_picks}_stratified_selection.csv"
    else:
        logger.info("[MODE] Cutoff => sub-bins inside the user-specified range")
        sel_path = cutoff_selection(
            df_path_only,  args.n_path,    "path_only",
            args.path_cutoff, args.cons_cutoff, args.both_cutoff
        )
        sel_cons = cutoff_selection(
            df_cons_only, args.n_consult, "consult_only",
            args.path_cutoff, args.cons_cutoff, args.both_cutoff
        )
        sel_both = cutoff_selection(
            df_both,      args.n_both,    "path_cons",
            args.path_cutoff, args.cons_cutoff, args.both_cutoff
        )
        outname = f"final_{total_picks}_cutoff_selection.csv"

    df_final = pd.concat([sel_path, sel_cons, sel_both], ignore_index=True).drop_duplicates(subset=["patient_id"])
    logger.info(f"Final => {len(df_final)} total => requested {args.n_path}+{args.n_consult}+{args.n_both}={total_picks}")
    out_csv = os.path.join(args.output_dir, outname)
    df_final.to_csv(out_csv, index=False)
    logger.info(f"Saved final selection => {out_csv}")

if __name__ == "__main__":
    main()


# Stratified mode - random stratified selection from low, medium, and high 'Not inferred' occurences with no cutoffs:
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/sample_selection.py \
#   --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
#   --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
#   --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
#   --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection" \
#   --mode stratified \
#   --n_path 5 \
#   --n_consult 18 \
#   --n_both 27
#   --both_cutoff 10 --path_cutoff 15 --cons_cutoff 15


# Cutoff (with Sub-binning) Mode - - random stratified selection from low, medium, and high 'Not inferred' occurences with cutoffs
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/sample_selection.py \
#   --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
#   --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
#   --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
#   --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection" \
#   --mode cutoff \
#   --n_path 5 \
#   --n_consult 18 \
#   --n_both 27 \
#   --both_cutoff 10 \
#   --path_cutoff 20 \
#   --cons_cutoff 11

# INFO: [MODE] Cutoff => sub-bins inside the user-specified range
# INFO: [cutoff+sub-bins] path_only: requested 5, got 5 out of 19 possible
# INFO: [cutoff+sub-bins] consult_only: requested 18, got 18 out of 133 possible
# INFO: [cutoff+sub-bins] path_cons: requested 27, got 27 out of 260 possible
# INFO: Final => 50 total => requested 5+18+27=50
# INFO: Saved final selection => /Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection/final_50_cutoff_selection.csv