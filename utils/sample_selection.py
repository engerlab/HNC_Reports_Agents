#!/usr/bin/env python3
"""
select_stratified_cases.py

1. Loads three categories of patient IDs:
   - path_only_ids.csv
   - consult_only_ids.csv
   - both_ids.csv

2. Loads processing_times.csv and merges the following fields:
   - process_time_ms
   - num_input_characters
   - num_input_tokens

3. For each ID, opens the path_consult_reports_summary.txt to count how many lines contain "Not inferred" (not_inferred_count).

4. Saves a CSV for each category containing:
   [patient_id, process_time_ms, num_input_characters, num_input_tokens, not_inferred_count]

5. Creates scatter plots of num_input_tokens vs not_inferred_count.

6. Performs a stratified selection based on not_inferred_count buckets (low, medium, high).

USAGE (example):
  python select_stratified_cases.py \
    --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
    --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
    --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
    --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
    --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection" 
"""

import os
import re
import argparse
import logging
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_ids(csv_path):
    """
    Loads a CSV file containing a column named 'patient_id'
    and returns a set of those IDs as strings.
    """
    df = pd.read_csv(csv_path)
    return set(map(str, df["patient_id"].tolist()))

def count_not_inferred(summary_file):
    """
    Count how many lines in the summary file contain 'Not inferred'
    If the file doesn't exist, return None (or 30 if you want to treat missing as all 'Not inferred').
    """
    if not os.path.isfile(summary_file):
        # Return None or 30, depending on your preference
        return None
    c = 0
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Not inferred" in line:
                c += 1
    return c

def create_scatter_plot(df, category_name, output_dir):
    """
    Creates a scatter plot of num_input_tokens vs not_inferred_count,
    saving to output_dir as {category_name}_scatter.png.
    """
    plt.figure(figsize=(6,4))
    plt.scatter(df["num_input_tokens"], df["not_inferred_count"], alpha=0.6)
    plt.xlabel("num_input_tokens")
    plt.ylabel("not_inferred_count")
    plt.title(f"{category_name}: tokens vs. not_inferred")
    out_path = os.path.join(output_dir, f"{category_name}_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Scatter plot saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Base directory for the run, containing processing_times.csv and text_summaries/path_consult_reports")
    parser.add_argument("--path_only", required=True,
                        help="CSV with column 'patient_id' for pathology-only")
    parser.add_argument("--consult_only", required=True,
                        help="CSV with column 'patient_id' for consult-only")
    parser.add_argument("--both", required=True,
                        help="CSV with column 'patient_id' for both path+consult")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for CSVs and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the ID sets
    path_only_ids = load_ids(args.path_only)
    consult_only_ids = load_ids(args.consult_only)
    both_ids = load_ids(args.both)

    # 2) Load processing_times.csv
    processing_csv = os.path.join(args.base_dir, "processing_times.csv")
    df_times = pd.read_csv(processing_csv)
    # Strip .txt from 'file' => becomes 'patient_id'
    df_times["patient_id"] = df_times["file"].apply(lambda x: os.path.splitext(x)[0])

    # We only want rows where report_type=="path_consult_reports"
    # (assuming that's where your summary is)
    df_times = df_times[df_times["report_type"]=="path_consult_reports"].copy()
    logger.info(f"Filtering for path_consult_reports => {len(df_times)} rows in processing_times.csv remain")

    # Directory with text_summaries => base_dir + text_summaries/path_consult_reports
    summaries_dir = os.path.join(args.base_dir, "text_summaries", "path_consult_reports")

    def build_category_df(cat_name, cat_ids):
        """
        For each ID in cat_ids, gather:
          - process_time_ms
          - num_input_characters
          - num_input_tokens
          - not_inferred_count
        Then return a DataFrame with columns:
         [patient_id, process_time_ms, num_input_characters, num_input_tokens, not_inferred_count].
        """
        subset = df_times[df_times["patient_id"].isin(cat_ids)].copy()
        not_inferred_counts = []
        for idx, row in subset.iterrows():
            pid = str(row["patient_id"])
            summary_file = os.path.join(summaries_dir, pid, "path_consult_reports_summary.txt")
            c = count_not_inferred(summary_file)
            not_inferred_counts.append(c if c is not None else np.nan)
        subset["not_inferred_count"] = not_inferred_counts
        # Sort columns for neatness
        subset = subset[["patient_id","process_time_ms","num_input_characters","num_input_tokens","not_inferred_count"]]
        # Sort by not_inferred_count ascending
        subset.sort_values(by="not_inferred_count", inplace=True, na_position="last")
        return subset

    # Build the DataFrames
    df_path_only = build_category_df("path_only", path_only_ids)
    df_cons_only = build_category_df("consult_only", consult_only_ids)
    df_both      = build_category_df("both", both_ids)

    # 3) Save each category as CSV
    path_only_csv = os.path.join(args.output_dir, "path_only_stats.csv")
    df_path_only.to_csv(path_only_csv, index=False)
    logger.info(f"Saved {len(df_path_only)} path-only stats to {path_only_csv}")

    cons_only_csv = os.path.join(args.output_dir, "consult_only_stats.csv")
    df_cons_only.to_csv(cons_only_csv, index=False)
    logger.info(f"Saved {len(df_cons_only)} consult-only stats to {cons_only_csv}")

    both_csv = os.path.join(args.output_dir, "both_stats.csv")
    df_both.to_csv(both_csv, index=False)
    logger.info(f"Saved {len(df_both)} both stats to {both_csv}")

    # 4) Create scatter plots
    create_scatter_plot(df_path_only, "path_only", args.output_dir)
    create_scatter_plot(df_cons_only, "consult_only", args.output_dir)
    create_scatter_plot(df_both, "both", args.output_dir)

    # 5) Stratification: for example, define bins for not_inferred_count
    # Let's define low: 0-10, med: 11-20, high: 21-30
    # Then pick e.g. 1/3 from each bin. You can adjust as you prefer.

    def stratify_by_not_inferred(df, category_label, sample_size):
        """ 
        df => has columns [patient_id, ..., not_inferred_count]
        We define bins: 0-10 => low, 11-20 => med, 21-30 => high
        Then sample proportionally from each bin. 
        For demonstration, let's do 1/3 from each bin if they exist.
        If some bin doesn't have enough rows, we handle that gracefully.
        """
        bins = [(0,10), (11,20), (21,30)]
        out_frames = []
        leftover = sample_size
        for bin_range in bins:
            low, high = bin_range
            sub = df[(df["not_inferred_count"]>=low) & (df["not_inferred_count"]<=high)].copy()
            if len(sub)==0:
                continue
            # We'll do an even distribution => sample_size/3 from each bin
            portion = sample_size // 3
            # If leftover isn't divisible, the last bin can pick up remainder
            # We'll do a random sample, ignoring if portion>len(sub)
            # If portion> len(sub), use len(sub).
            pick = min(portion, len(sub))
            sample_df = sub.sample(pick, random_state=42)
            leftover -= pick
            out_frames.append(sample_df)

        # If leftover>0 for some reason (some bins smaller?), we might just fill from entire df
        if leftover>0:
            # We can pick leftover from entire df that hasn't been chosen yet
            already_chosen_ids = set()
            for tmp in out_frames:
                already_chosen_ids.update(tmp["patient_id"].tolist())
            sub_remaining = df[~df["patient_id"].isin(already_chosen_ids)]
            if len(sub_remaining)>0:
                leftover_pick = min(leftover, len(sub_remaining))
                out_frames.append(sub_remaining.sample(leftover_pick, random_state=123))

        final_df = pd.concat(out_frames).drop_duplicates(subset=["patient_id"])
        final_df = final_df.sort_values(by="not_inferred_count")
        logger.info(f"{category_label}: Requested {sample_size}, got {len(final_df)} after stratification.")
        return final_df

    # Let's define how many we want from each category
    # Example: path_only=5, consult_only=18, both=27 => total=50
    pick_sizes = {"path_only":5, "consult_only":18, "both":27}

    df_path_sel = stratify_by_not_inferred(df_path_only, "path_only", pick_sizes["path_only"])
    df_cons_sel = stratify_by_not_inferred(df_cons_only, "consult_only", pick_sizes["consult_only"])
    df_both_sel = stratify_by_not_inferred(df_both,      "both",        pick_sizes["both"])

    # Combine into final
    df_final = pd.concat([df_path_sel, df_cons_sel, df_both_sel], ignore_index=True)
    final_csv = os.path.join(args.output_dir, "final_50_stratified_selection.csv")
    df_final.to_csv(final_csv, index=False)
    logger.info(f"Saved final stratified selection => {len(df_final)} cases => {final_csv}")

if __name__=="__main__":
    main()


# Use examples:
#   python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/sample_selection.py \
#     --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
#     --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
#     --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
#     --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
#     --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection" 