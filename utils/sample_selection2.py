#!/usr/bin/env python3
"""
sample_selection_exp14.py

Selects a subset of patient IDs from path_only, consult_only, and both categories,
using the new Exp14 directory that contains:
  - final_merged_processing_times.csv  (only file, report_type, process_time_ms)
  - precompute_tokens_mode2.csv        (columns like extr_total_needed, cot_total_needed, total_needed, etc.)
  - text_summaries/mode2_combined_twostep/<pid>/path_consult_reports_summary.txt
    for each .txt that was processed
  - the num_input_tokens in the plots were the max of extr_total_needed and cot_total_needed (since this was Experiment mode 2, the max of them will be the max token needed per query.)

Adds:
  1) 'category' column in final CSV: path_only, consult_only, path_cons.
  2) --n_path, --n_consult, --n_both arguments to specify how many from each category.
  3) --both_cutoff, --path_cutoff, --cons_cutoff to control numeric thresholds in cutoff mode
     (like the old script: "cutoff" or "stratified" approach).
  4) We define a single "num_input_tokens" from precompute_tokens_mode2.csv
     - For a two-step approach, we take max(extr_total_needed, cot_total_needed)
     - If only single-step columns exist (like total_needed), we use that
  5) This script then merges that with final_merged_processing_times.csv on 'file'.

Usage Examples (Stratified mode):
  python sample_selection_exp14.py \
    --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14" \
    --path_only "/Data/.../path_only_ids.csv" \
    --consult_only "/Data/.../consult_only_ids.csv" \
    --both "/Data/.../both_ids.csv" \
    --output_dir "/Data/.../CaseSelection/Exp14" \
    --mode stratified \
    --n_path 5 --n_consult 18 --n_both 27

Or (Cutoff mode):
  python sample_selection_exp14.py \
    --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14" \
    --path_only "/Data/.../path_only_ids.csv" \
    --consult_only "/Data/.../consult_only_ids.csv" \
    --both "/Data/.../both_ids.csv" \
    --output_dir "/Data/.../CaseSelection/Exp14" \
    --mode cutoff \
    --n_path 5 --n_consult 18 --n_both 27 \
    --both_cutoff 8 --path_cutoff 15 --cons_cutoff 10
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


###############################################################################
# 1) ID loading for path_only, consult_only, both
###############################################################################
def load_ids(csv_path):
    df = pd.read_csv(csv_path)
    # Expect a 'patient_id' column
    return set(map(str, df["patient_id"].tolist()))

###############################################################################
# 2) Count "not inferred" lines from the final summary
###############################################################################
def count_not_inferred(summary_file):
    if not os.path.isfile(summary_file):
        return None
    c = 0
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            if "not inferred" in line.strip().lower():
                c += 1
    return c

###############################################################################
# 3) For scatter plots
###############################################################################
def create_scatter_plot(df, category_name, output_dir):
    df2 = df.dropna(subset=["num_input_tokens","not_inferred_count"])
    if df2.empty:
        logger.warning(f"{category_name}: no data for scatter plot.")
        return
    plt.figure(figsize=(6,4))
    plt.scatter(df2["num_input_tokens"], df2["not_inferred_count"], alpha=0.6)
    plt.xlabel("Number of Max Input Tokens", fontsize = 18)
    plt.ylabel("Number of 'Not Inferred' Fields", fontsize = 18)
    plt.title(f"{category_name}", fontsize = 20)
    out_path = os.path.join(output_dir, f"{category_name}_scatter.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Scatter plot saved: {out_path}")

def create_jointplot(df, category_name, output_dir):
    df2 = df.dropna(subset=["num_input_tokens","not_inferred_count"]).copy()
    if df2.empty:
        logger.warning(f"{category_name}: no data for jointplot.")
        return

    def categorize_not_inferred(x):
        if x < 10:
            return "low"
        elif x <= 20:
            return "medium"
        else:
            return "high"

    df2["bin_not_inferred"] = df2["not_inferred_count"].apply(categorize_not_inferred)
    sns.set_context("talk", font_scale=1.05)  
    # "talk" context and "font_scale=1.3" are just examples; you can use "paper", "poster", etc.

    g = sns.jointplot(
        data=df2,
        x="num_input_tokens",
        y="not_inferred_count",
        hue="bin_not_inferred",
        alpha=0.6
    )
    # 3) Adjust axis labels, tick labels, title, etc.
    g.ax_joint.set_xlabel("Number of Max Input Tokens", fontsize=18)
    g.ax_joint.set_ylabel("Number of 'Not Inferred' Fields", fontsize=18)
    g.ax_joint.tick_params(axis='both', labelsize=15)

    # If you have a legend, you can also adjust its font size:
    if g.ax_joint.get_legend() is not None:
        plt.setp(g.ax_joint.get_legend().get_texts(), fontsize=12)  # legend labels
        plt.setp(g.ax_joint.get_legend().get_title(), fontsize=13)  # legend title

    # g.fig.suptitle(f"{category_name} JointPlot by not_inferred bins", y=1, fontsize=20)
    g.fig.suptitle(f"{category_name}", y=1, fontsize=20)
    out_path = os.path.join(output_dir, f"{category_name}_joint.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Jointplot saved: {out_path}")

###############################################################################
# 4) category DataFrame builder
###############################################################################
def build_category_df(cat_name, cat_ids, df_times, summaries_dir):
    """
    For each ID in cat_ids => we keep only rows from df_times if patient_id is in cat_ids.
      columns => [patient_id, category, process_time_ms, num_input_tokens, not_inferred_count]
    """
    sub = df_times[df_times["patient_id"].isin(cat_ids)].copy()
    sub["category"] = cat_name

    # For not_inferred_count, we look up the final text summary
    not_inferred_counts = []
    for idx, row in sub.iterrows():
        pid = str(row["patient_id"])
        # Summaries should be in text_summaries/mode2_combined_twostep/<pid>/path_consult_reports_summary.txt
        summary_file = os.path.join(summaries_dir, pid, "path_consult_reports_summary.txt")
        c = count_not_inferred(summary_file)
        not_inferred_counts.append(c if c is not None else np.nan)

    sub["not_inferred_count"] = not_inferred_counts

    # We'll keep these columns if they exist
    # to avoid KeyErrors, let's reindex
    final_cols = [
        "patient_id","category","process_time_ms","num_input_tokens","not_inferred_count"
    ]
    # ensure these columns exist (others are dropped)
    for c in final_cols:
        if c not in sub.columns:
            sub[c] = np.nan

    sub = sub[final_cols]
    sub.sort_values(by="not_inferred_count", inplace=True, na_position="last")
    return sub

###############################################################################
# 5) Stratify / Cutoff
###############################################################################
def stratify_by_not_inferred(df, sample_size, category_label):
    """
    Bins => 0-10 => low, 11-20 => med, 21-30 => high
    We pick ~ sample_size/3 from each bin if possible.
    """
    bins = [(0,10), (11,20), (21,30)]
    out_frames = []
    leftover = sample_size
    for (low, high) in bins:
        sub = df[(df["not_inferred_count"]>=low) & (df["not_inferred_count"]<=high)].copy()
        if sub.empty:
            continue
        portion = sample_size // 3
        pick = min(portion, len(sub))
        sample_df = sub.sample(pick, random_state=42)
        leftover -= pick
        out_frames.append(sample_df)

    # if leftover>0, fill from remainder
    if leftover>0:
        chosen_ids = set()
        for chunk in out_frames:
            chosen_ids.update(chunk["patient_id"].tolist())
        remain = df[~df["patient_id"].isin(chosen_ids)]
        if not remain.empty:
            leftover_pick = min(leftover, len(remain))
            out_frames.append(remain.sample(leftover_pick, random_state=123))

    final_df = pd.concat(out_frames, ignore_index=True).drop_duplicates(subset=["patient_id"])
    final_df.sort_values(by="not_inferred_count", inplace=True)
    logger.info(f"[stratify] {category_label}: requested {sample_size}, got {len(final_df)}")
    return final_df

def define_subbins(start_val, end_val, n_subbins=2):
    if start_val> end_val:
        return []
    total_range = end_val - start_val + 1
    step = total_range // n_subbins
    bins = []
    current_low = start_val
    for i in range(n_subbins):
        if i == n_subbins-1:
            current_high = end_val
        else:
            current_high = current_low + step - 1
        if current_high < current_low:
            break
        bins.append((current_low, current_high))
        current_low = current_high+1
    return bins

def cutoff_selection(df, sample_size, category_type, path_cutoff, cons_cutoff, both_cutoff):
    """
    If category is path_cons => both_cutoff
       path_only  => path_cutoff
       consult_only => cons_cutoff
    Then sub-binning inside that range.
    """
    if category_type == "path_cons":
        limit = both_cutoff
    elif category_type == "path_only":
        limit = path_cutoff
    else:
        limit = cons_cutoff

    df_filt = df[df["not_inferred_count"] < limit]
    # define sub-bins
    sub_bins = define_subbins(0, limit-1, n_subbins=3)

    out_frames = []
    leftover = sample_size
    for (low, high) in sub_bins:
        sub = df_filt[(df_filt["not_inferred_count"]>=low) & (df_filt["not_inferred_count"]<=high)].copy()
        if sub.empty:
            continue
        portion = sample_size // len(sub_bins)
        pick = min(portion, len(sub))
        chunk = sub.sample(pick, random_state=42)
        leftover -= pick
        out_frames.append(chunk)

    if leftover>0:
        chosen_ids = set()
        for chunk in out_frames:
            chosen_ids.update(chunk["patient_id"].tolist())
        remain = df_filt[~df_filt["patient_id"].isin(chosen_ids)]
        if not remain.empty:
            leftover_pick = min(leftover, len(remain))
            out_frames.append(remain.sample(leftover_pick, random_state=123))

    final_df = pd.concat(out_frames, ignore_index=True).drop_duplicates(subset=["patient_id"])
    final_df.sort_values(by="not_inferred_count", inplace=True)
    logger.info(f"[cutoff+sub-bins] {category_type}: requested {sample_size}, got {len(final_df)} out of {len(df_filt)} possible")
    return final_df


###############################################################################
# 6) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser("Sample Selection for Exp14 with merged + precompute tokens")
    parser.add_argument("--base_dir", required=True,
                        help="Folder with final_merged_processing_times.csv, precompute_tokens_mode2.csv, text_summaries/mode2_combined_twostep.")
    parser.add_argument("--path_only", required=True,
                        help="CSV with 'patient_id' for path-only patients.")
    parser.add_argument("--consult_only", required=True,
                        help="CSV with 'patient_id' for consult-only patients.")
    parser.add_argument("--both", required=True,
                        help="CSV with 'patient_id' for path+consult patients.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to place the final stats, scatter plots, etc.")
    parser.add_argument("--mode", choices=["stratified","cutoff"], default="stratified",
                        help="Which selection approach to use.")
    parser.add_argument("--n_path", type=int, default=5, help="How many path_only to pick.")
    parser.add_argument("--n_consult", type=int, default=18, help="How many consult_only to pick.")
    parser.add_argument("--n_both", type=int, default=27, help="How many path_cons to pick.")
    parser.add_argument("--both_cutoff", type=int, default=10, help="Cutoff for path_cons category if mode=cutoff.")
    parser.add_argument("--path_cutoff", type=int, default=15, help="Cutoff for path_only if mode=cutoff.")
    parser.add_argument("--cons_cutoff", type=int, default=15, help="Cutoff for consult_only if mode=cutoff.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the ID sets
    path_ids = load_ids(args.path_only)
    cons_ids = load_ids(args.consult_only)
    both_ids = load_ids(args.both)

    # 2) Load final_merged_processing_times.csv => only columns: file, report_type, process_time_ms
    merged_times_csv = os.path.join(args.base_dir, "final_merged_processing_times.csv")
    if not os.path.isfile(merged_times_csv):
        logger.error(f"No final_merged_processing_times.csv in {args.base_dir}")
        return
    df_times = pd.read_csv(merged_times_csv)

    # Convert 'file' => 'patient_id' (strip .txt)
    df_times["patient_id"] = df_times["file"].apply(lambda x: os.path.splitext(x)[0])

    # 3) Load precompute_tokens_mode2.csv => has extr_total_needed, cot_total_needed, or single-step total_needed
    precompute_csv = os.path.join(args.base_dir, "precompute_tokens_mode2.csv")
    if not os.path.isfile(precompute_csv):
        logger.warning(f"No precompute_tokens_mode2.csv found. Setting num_input_tokens=NaN for all.")
        df_times["num_input_tokens"] = np.nan
    else:
        df_pre = pd.read_csv(precompute_csv)  # columns: file, extr_total_needed, ...
        df_pre["file"] = df_pre["file"].astype(str)
        # We'll define a single "token_needed" column
        # If two-step => extr_total_needed + cot_total_needed exist, we might do max of those
        # if single-step => total_needed might exist
        def compute_num_tokens(row):
            # Check for a multi-step scenario
            c_extr = row.get("extr_total_needed", np.nan)
            c_cot  = row.get("cot_total_needed", np.nan)
            c_single = row.get("total_needed", np.nan)

            # If extr/cot exist => take max
            if not pd.isna(c_extr) or not pd.isna(c_cot):
                return np.nanmax([c_extr, c_cot])
            else:
                return c_single

        # Build a single column
        df_pre["num_input_tokens"] = df_pre.apply(compute_num_tokens, axis=1)
        # Merge into df_times on 'file'
        df_times = df_times.merge(df_pre[["file","num_input_tokens"]], on="file", how="left")

    # So now df_times has columns:
    #   [file, report_type, process_time_ms, patient_id, num_input_tokens]

    # 4) Summaries folder => text_summaries/mode2_combined_twostep
    summaries_dir = os.path.join(args.base_dir, "text_summaries", "mode2_combined_twostep")

    # 5) Build DataFrames for each category
    df_path_only = build_category_df("path_only",   path_ids,  df_times, summaries_dir)
    df_cons_only = build_category_df("consult_only",cons_ids,  df_times, summaries_dir)
    df_both      = build_category_df("path_cons",   both_ids,  df_times, summaries_dir)

    # Save them for reference
    df_path_only.to_csv(os.path.join(args.output_dir,"path_only_stats.csv"), index=False)
    df_cons_only.to_csv(os.path.join(args.output_dir,"consult_only_stats.csv"), index=False)
    df_both     .to_csv(os.path.join(args.output_dir,"both_stats.csv"), index=False)

    # 6) Make scatter + joint plots
    def make_plots(df, cat_name):
        create_scatter_plot(df, cat_name, args.output_dir)
        create_jointplot(df, cat_name, args.output_dir)

    make_plots(df_path_only, "path_only")
    make_plots(df_cons_only, "consult_only")
    make_plots(df_both,      "path_cons")

    # 7) Final selection
    total_picks = args.n_path + args.n_consult + args.n_both
    logger.info(f"Selecting total={total_picks} with mode={args.mode}")

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


# Usage example 
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/sample_selection2.py \
#   --base_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14" \
#   --path_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/path_only_ids.csv" \
#   --consult_only "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/consult_only_ids.csv" \
#   --both "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/both_ids.csv" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection/Exp14" \
#   --mode stratified \
#   --n_path 5 --n_consult 18 --n_both 27 \
#   --mode cutoff --both_cutoff 8 --path_cutoff 20 --cons_cutoff 10