#!/usr/bin/env python3
"""
path_cons_stats.py

Collects basic stats for pathology vs. consultation note patient IDs, 
using the same directory logic as hnc_reports_agent4.py:

  - PathologyReports/ under --input_dir
  - ConsultRedacted/ under --input_dir

Then it prints (and optionally saves):
  - # of unique pathology report IDs
  - # of unique consultation note IDs
  - # of total unique patient IDs (the union)
  - # of overlap (the intersection, i.e. # patients with both path+consult)
  - # of pathology-only and consultation-only patient IDs

Additionally, if the optional argument --csv_output_dir is provided, 
the script will output six CSV files (one per category) into that directory.

Usage:
    python path_cons_stats.py \
        --input_dir "/media/yujing/One Touch3/HNC_Reports" \
        --stats_output "/path/to/stats_output.txt" \
        --csv_output_dir "/path/to/csv_output_dir"
"""

import os
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def gather_ids_from_folder(folder_path: str) -> set:
    """
    Returns a set of patient IDs (filenames without extension) for .txt files in folder_path.
    If folder doesn't exist, logs a warning and returns an empty set.
    """
    ids = set()
    if not os.path.isdir(folder_path):
        logger.warning(f"Folder not found: {folder_path}")
        return ids
    
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            pid = os.path.splitext(fname)[0]
            ids.add(pid)
    return ids

def main():
    parser = argparse.ArgumentParser(
        description="Compute stats for pathology vs. consultation notes in the same structure used by hnc_reports_agent4.py."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Parent directory with PathologyReports/ and ConsultRedacted/ subfolders.")
    parser.add_argument("--stats_output", default="",
                        help="(Optional) Path to a text file where summary results will be saved.")
    parser.add_argument("--csv_output_dir", default="",
                        help="(Optional) Directory where individual CSV files for each category will be saved.")
    args = parser.parse_args()

    # Define folder paths
    pathology_folder = os.path.join(args.input_dir, "PathologyReports")
    consult_folder   = os.path.join(args.input_dir, "ConsultRedacted")

    # Gather patient IDs
    path_ids = gather_ids_from_folder(pathology_folder)
    cons_ids = gather_ids_from_folder(consult_folder)

    # Compute basic stats
    num_path = len(path_ids)
    num_cons = len(cons_ids)
    union_ids = path_ids.union(cons_ids)
    intersect_ids = path_ids.intersection(cons_ids)

    num_total = len(union_ids)
    num_both = len(intersect_ids)
    path_only = path_ids - cons_ids
    cons_only = cons_ids - path_ids

    # Log stats
    logger.info(f"Pathology reports: {num_path}")
    logger.info(f"Consultation notes: {num_cons}")
    logger.info(f"Total unique patients: {num_total}")
    logger.info(f"Patients with both path+consult: {num_both}")
    logger.info(f"Path-only patient count: {len(path_only)}")
    logger.info(f"Consult-only patient count: {len(cons_only)}")

    # Optionally save summary stats to a text file
    if args.stats_output:
        result_lines = [
            f"Pathology reports: {num_path}",
            f"Consultation notes: {num_cons}",
            f"Total unique patients: {num_total}",
            f"Patients with both path+consult: {num_both}",
            f"Path-only patient count: {len(path_only)}",
            f"Consult-only patient count: {len(cons_only)}",
        ]
        try:
            with open(args.stats_output, "w", encoding="utf-8") as outf:
                outf.write("\n".join(result_lines) + "\n")
            logger.info(f"Stats saved to {os.path.abspath(args.stats_output)}")
        except Exception as e:
            logger.error(f"Failed to save stats output: {e}")

    # If CSV output directory is provided, save individual CSV files
    if args.csv_output_dir:
        try:
            os.makedirs(args.csv_output_dir, exist_ok=True)
            csv_files = {
                "pathology_ids.csv": sorted(list(path_ids)),
                "consultation_ids.csv": sorted(list(cons_ids)),
                "total_unique_ids.csv": sorted(list(union_ids)),
                "both_ids.csv": sorted(list(intersect_ids)),
                "path_only_ids.csv": sorted(list(path_only)),
                "consult_only_ids.csv": sorted(list(cons_only))
            }
            for filename, ids_list in csv_files.items():
                csv_path = os.path.join(args.csv_output_dir, filename)
                pd.DataFrame({"patient_id": ids_list}).to_csv(csv_path, index=False)
                logger.info(f"Saved {filename} to {os.path.abspath(csv_path)}")
        except Exception as e:
            logger.error(f"Failed to save CSV files: {e}")

if __name__ == "__main__":
    main()


# For obtaining stats on the HNC reports directory for numbers on Venn Diagram
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/path_cons_stats.py \
#     --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#     --stats_output "/tmp/stats_output.txt"

# Save specifc patient IDs to CSV files for each category 
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/path_cons_stats.py \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --stats_output "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/stats_output.txt" \
#   --csv_output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats"

