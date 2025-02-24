#!/usr/bin/env python3
"""
select_stratified_cases.py

Enhanced version: in cutoff mode, we still sub-stratify within the allowable range to reflect 
the distribution of not_inferred_count. 

1) Reads path_only, consult_only, both IDs. 
2) Merges with processing_times.csv -> 
   keep only report_type=="path_consult_reports". 
3) For each ID, count "Not inferred" from the summary text. 
4) Saves CSV for each category. 
5) Creates scatter + jointplot. 
6) Two modes for final selection:
   - --mode stratified => same old approach (bins 0-10, 11-20, 21-30). 
   - --mode cutoff => 
       - both < 10 
       - path < 15 
       - consult <15
     Then we do sub-binning within that range 
     e.g. path < 15 => sub-bins might be [0-5, 6-10, 11-14], etc. 
     That ensures we reflect the distribution inside the cutoff as well.

Usage example:
  python select_stratified_cases.py \
    --base_dir "/Data/.../Exp13" \
    --path_only "/Data/.../path_only_ids.csv" \
    --consult_only "/Data/.../consult_only_ids.csv" \
    --both "/Data/.../both_ids.csv" \
    --output_dir "/Data/.../case_selection_Exp13" \
    --mode cutoff
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

# We define how many cases to pick from each category:
SAMPLE_SIZES = {
    "path_only": 5,
    "consult_only": 18,
    "both": 27
}

def load_ids(csv_path):
    df = pd.read_csv(csv_path)
    return set(map(str, df["patient_id"].tolist()))

def count_not_inferred(summary_file):
    if not os.path.isfile(summary_file):
        return None
    c = 0
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            if "Not inferred" in line.strip():
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Scatter plot saved: {out_path}")

def create_jointplot(df, category_name, output_dir):
    """
    Seaborn jointplot with color-coded bins for not_inferred_count:
      - low: <10
      - medium: 10-20
      - high: >20
    """
    df2 = df.dropna(subset=["num_input_tokens","not_inferred_count"]).copy()

    def categorize_not_inferred(x):
        if x < 10:
            return "low"
        elif x <= 20:
            return "medium"
        else:
            return "high"

    df2["bin_not_inferred"] = df2["not_inferred_count"].apply(categorize_not_inferred)

    g = sns.jointplot(
        data=df2,
        x="num_input_tokens",
        y="not_inferred_count",
        hue="bin_not_inferred",
        alpha=0.6
    )
    g.fig.suptitle(f"{category_name} JointPlot by not_inferred bins", y=1.02)
    out_path = os.path.join(output_dir, f"{category_name}_joint.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Jointplot saved: {out_path}")

def build_category_df(cat_ids, df_times, summaries_dir):
    """
    For each ID in cat_ids => 
      [patient_id, process_time_ms, num_input_characters, num_input_tokens, not_inferred_count]
    """
    subset = df_times[df_times["patient_id"].isin(cat_ids)].copy()
    not_inferred_counts = []
    for idx, row in subset.iterrows():
        pid = str(row["patient_id"])
        summary_file = os.path.join(summaries_dir, pid, "path_consult_reports_summary.txt")
        c = count_not_inferred(summary_file)
        not_inferred_counts.append(c if c is not None else np.nan)
    subset["not_inferred_count"] = not_inferred_counts
    subset = subset[[
        "patient_id","process_time_ms","num_input_characters","num_input_tokens","not_inferred_count"
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

    final_df = pd.concat(out_frames).drop_duplicates(subset=["patient_id"])
    final_df.sort_values(by="not_inferred_count", inplace=True)
    logger.info(f"[stratify] {category_label}: requested {sample_size}, got {len(final_df)}")
    return final_df

def cutoff_selection(df, sample_size, category_type):
    """
    Now with sub-binning inside the allowable range:
     - if category_type=="both" => filter not_inferred_count<10
       then sub-bins: e.g. (0-5) => low, (6-9) => high
     - if category_type=="path_only" => filter <15 => sub-bins: (0-5), (6-10), (11-14)
     - if category_type=="consult_only" => same as path_only (<15)...

    Then we do random sampling across sub-bins in proportion to sample_size/ #bins.
    """
    if category_type == "both":
        # e.g. we keep rows <10 => sub-bins: 0-5, 6-9
        df_filt = df[df["not_inferred_count"]<10]
        bins = [(0,5), (6,9)]
    elif category_type == "path_only":
        # <15 => define e.g. bins: 0-5, 6-10, 11-14
        df_filt = df[df["not_inferred_count"]<15]
        bins = [(0,5), (6,10), (11,14)]
    else: # consult_only
        # same as path_only => <15 => bins
        df_filt = df[df["not_inferred_count"]<15]
        bins = [(0,5), (6,10), (11,14)]

    out_frames = []
    leftover = sample_size
    # sub-binning
    for bin_range in bins:
        low, high = bin_range
        sub = df_filt[(df_filt["not_inferred_count"]>=low) & (df_filt["not_inferred_count"]<=high)].copy()
        if len(sub)==0:
            continue
        portion = sample_size // len(bins)
        pick = min(portion, len(sub))
        sample_df = sub.sample(pick, random_state=42)
        leftover -= pick
        out_frames.append(sample_df)

    if leftover>0:
        # fill from the entire df_filt not chosen yet
        chosen_ids = set()
        for tmp in out_frames:
            chosen_ids.update(tmp["patient_id"].tolist())
        sub_remaining = df_filt[~df_filt["patient_id"].isin(chosen_ids)]
        if len(sub_remaining)>0:
            leftover_pick = min(leftover, len(sub_remaining))
            out_frames.append(sub_remaining.sample(leftover_pick, random_state=123))

    final_df = pd.concat(out_frames).drop_duplicates(subset=["patient_id"])
    final_df.sort_values(by="not_inferred_count", inplace=True)
    logger.info(f"[cutoff+sub-bins] {category_type}: requested {sample_size}, got {len(final_df)} out of {len(df_filt)} possible")
    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--path_only", required=True)
    parser.add_argument("--consult_only", required=True)
    parser.add_argument("--both", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["stratified","cutoff"], default="stratified",
                        help="Which selection approach to use. 'stratified' or 'cutoff' with sub-binning.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load category IDs
    path_only_ids = load_ids(args.path_only)
    consult_only_ids = load_ids(args.consult_only)
    both_ids = load_ids(args.both)

    # 2) Load processing_times
    df_times = pd.read_csv(os.path.join(args.base_dir, "processing_times.csv"))
    df_times = df_times[df_times["report_type"]=="path_consult_reports"].copy()
    # patient_id = file w/o .txt
    df_times["patient_id"] = df_times["file"].apply(lambda x: os.path.splitext(x)[0])
    summaries_dir = os.path.join(args.base_dir, "text_summaries", "path_consult_reports")

    # Build each category DataFrame
    df_path_only = build_category_df(path_only_ids, df_times, summaries_dir)
    df_cons_only = build_category_df(consult_only_ids, df_times, summaries_dir)
    df_both      = build_category_df(both_ids,       df_times, summaries_dir)

    # Save them
    df_path_only.to_csv(os.path.join(args.output_dir,"path_only_stats.csv"), index=False)
    df_cons_only.to_csv(os.path.join(args.output_dir,"consult_only_stats.csv"), index=False)
    df_both     .to_csv(os.path.join(args.output_dir,"both_stats.csv"), index=False)

    # Plots
    def make_plots(df, cat_name):
        create_scatter_plot(df, cat_name, args.output_dir)
        create_jointplot(df, cat_name, args.output_dir)

    make_plots(df_path_only, "path_only")
    make_plots(df_cons_only, "consult_only")
    make_plots(df_both,      "both")

    # 3) Final selection
    pick_sizes = {
        "path_only": 5,
        "consult_only": 18,
        "both": 27
    }

    if args.mode=="stratified":
        logger.info("Using STRATIFIED mode => bins 0-10, 11-20, 21-30")
        df_p_sel = stratify_by_not_inferred(df_path_only,  pick_sizes["path_only"],    "path_only")
        df_c_sel = stratify_by_not_inferred(df_cons_only, pick_sizes["consult_only"], "consult_only")
        df_b_sel = stratify_by_not_inferred(df_both,      pick_sizes["both"],         "both")
        outname = "final_50_stratified_selection.csv"
    else:
        logger.info("Using CUTOFF mode => sub-binning inside the allowable range")
        df_p_sel = cutoff_selection(df_path_only,  pick_sizes["path_only"],    "path_only")
        df_c_sel = cutoff_selection(df_cons_only, pick_sizes["consult_only"], "consult_only")
        df_b_sel = cutoff_selection(df_both,      pick_sizes["both"],         "both")
        outname = "final_50_cutoff_selection.csv"

    df_final = pd.concat([df_p_sel, df_c_sel, df_b_sel], ignore_index=True)
    logger.info(f"Final => {len(df_final)} total => {pick_sizes['path_only']} + {pick_sizes['consult_only']} + {pick_sizes['both']}")
    df_final.to_csv(os.path.join(args.output_dir, outname), index=False)
    logger.info(f"Saved final => {outname}")

if __name__=="__main__":
    main()


# Stratified mode - random stratified selection from low, medium, and high 'Not inferred' occurences with no cutoffs:
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/sample_selection.py \
#   --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
#   --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
#   --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
#   --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection" \
#   --mode stratified

# Cutoff (with Sub-binning) Mode - - random stratified selection from low, medium, and high 'Not inferred' occurences with cutoffs
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/sample_selection.py \
#   --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
#   --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
#   --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
#   --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection" \
#   --mode cutoff
