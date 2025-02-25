#!/usr/bin/env python3
"""
extract_fields_to_csv.py

Script that extracts specific fields from each patient's text summary
(path_consult_reports_summary.txt) and saves them into two CSVs:
  1) Path_Structured/p16_hpv.csv => columns [patient_id, p16_Status, HPV_Status]
     * sorted so p16_Status = Positive first, Negative second, else last
  2) Cons_Structured/cons_fields.csv => columns 
     [patient_id, Smoking_History, Pack_Years, Alcohol_Consumption, 
      Charlson_Comorbidity_Score, Karnofsky_Performance_Status, ECOG_Performance_Status]
     * sorted by patient_id.

Any missing fields => "NaN" for textual fields; or numeric NaN if the parse fails.

Usage Example:

    python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/extract_fields_to_csv.py \
      --input_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
      --id_list "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/total_unique_ids.csv" \
      --out_path_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results"

This will create or update two subfolders in --out_path_dir:
    1) Path_Structured/  => p16_hpv.csv
    2) Cons_Structured/  => cons_fields.csv
"""

import os
import math
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The fields we want to extract from the summary
PATH_FIELDS = ["p16_Status", "HPV_Status"]
CONS_FIELDS = [
    "Smoking_History",
    "Pack_Years",
    "Alcohol_Consumption",
    "Charlson_Comorbidity_Score",
    "Karnofsky_Performance_Status",
    "ECOG_Performance_Status"
]

def parse_line_for_field(summary_file, field_name):
    """
    Returns the string value following "field_name:" in the summary file.
    If not found or file missing => return None.
    """
    if not os.path.isfile(summary_file):
        return None
    
    prefix = field_name + ":"
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith(prefix):
                # e.g. "Karnofsky_Performance_Status: 80 [converted from ECOG=1]"
                val_str = line_stripped[len(prefix):].strip()
                return val_str
    return None

def parse_numeric_value(field_val):
    """
    For fields like "3 [some detail]" => we parse "3" as int.
    If parse fails or field_val is empty => return math.nan.
    """
    if not field_val:  # None or empty
        return math.nan
    # Split by whitespace, parse first chunk as int
    first_chunk = field_val.split()[0]
    try:
        return int(first_chunk)
    except ValueError:
        return math.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Base directory with text_summaries/path_consult_reports/<pid>/path_consult_reports_summary.txt")
    parser.add_argument("--id_list", required=True,
                        help="CSV listing all unique IDs (e.g., total_unique_ids.csv).")
    parser.add_argument("--out_path_dir", required=True,
                        help="Where to place Path_Structured/ and Cons_Structured/ subdirs + CSV outputs.")
    args = parser.parse_args()

    # 1) Load IDs
    df_ids = pd.read_csv(args.id_list)
    patient_ids = df_ids["patient_id"].astype(str).tolist()
    logger.info(f"Loaded {len(patient_ids)} patient IDs from {args.id_list}")

    # 2) Create subfolders
    path_subdir = os.path.join(args.out_path_dir, "Path_Structured")
    cons_subdir = os.path.join(args.out_path_dir, "Cons_Structured")
    os.makedirs(path_subdir, exist_ok=True)
    os.makedirs(cons_subdir, exist_ok=True)

    # 3) We'll gather lists of dicts => DataFrames
    path_records = []
    cons_records = []

    def get_summary_path(pid):
        return os.path.join(
            args.input_dir,
            "text_summaries",
            "path_consult_reports",
            pid,
            "path_consult_reports_summary.txt"
        )

    for pid in patient_ids:
        sfile = get_summary_path(pid)

        # Path fields
        val_p16 = parse_line_for_field(sfile, "p16_Status")
        val_hpv = parse_line_for_field(sfile, "HPV_Status")
        if val_p16 is None:  val_p16 = "NaN"
        if val_hpv is None:  val_hpv = "NaN"

        path_records.append({
            "patient_id": pid,
            "p16_Status": val_p16,
            "HPV_Status": val_hpv
        })

        # Cons fields
        val_smoking = parse_line_for_field(sfile, "Smoking_History")
        val_packyrs = parse_line_for_field(sfile, "Pack_Years")
        val_alcohol = parse_line_for_field(sfile, "Alcohol_Consumption")
        val_charlson = parse_line_for_field(sfile, "Charlson_Comorbidity_Score")
        val_kps = parse_line_for_field(sfile, "Karnofsky_Performance_Status")
        val_ecog = parse_line_for_field(sfile, "ECOG_Performance_Status")

        # Convert None => "NaN" for string fields
        def none2nan(s):
            return s if s is not None else "NaN"
        val_smoking = none2nan(val_smoking)
        val_packyrs = none2nan(val_packyrs)
        val_alcohol = none2nan(val_alcohol)

        # numeric fields => parse integer or NaN
        charlson_num = parse_numeric_value(val_charlson)
        kps_num = parse_numeric_value(val_kps)
        ecog_num = parse_numeric_value(val_ecog)

        cons_records.append({
            "patient_id": pid,
            "Smoking_History": val_smoking,
            "Pack_Years": val_packyrs,
            "Alcohol_Consumption": val_alcohol,
            "Charlson_Comorbidity_Score": charlson_num,
            "Karnofsky_Performance_Status": kps_num,
            "ECOG_Performance_Status": ecog_num
        })

    # 4) Build path DataFrame => sort by p16_Status (Positive first, Negative second, else last)
    df_path = pd.DataFrame(path_records)

    def custom_sort_order(val):
        val_low = val.lower()
        if "positive" in val_low:
            return 0
        elif "negative" in val_low:
            return 1
        else:
            return 2

    df_path["sort_key"] = df_path["p16_Status"].apply(custom_sort_order)
    df_path.sort_values(by=["sort_key", "patient_id"], inplace=True)
    df_path.drop(columns=["sort_key"], inplace=True)

    path_out = os.path.join(path_subdir, "p16_hpv.csv")
    df_path.to_csv(path_out, index=False)
    logger.info(f"Saved {len(df_path)} rows => {path_out}")

    # 5) Build cons DataFrame => sort by patient_id
    df_cons = pd.DataFrame(cons_records)
    df_cons.sort_values(by="patient_id", inplace=True)
    cons_out = os.path.join(cons_subdir, "cons_fields.csv")
    df_cons.to_csv(cons_out, index=False)
    logger.info(f"Saved {len(df_cons)} rows => {cons_out}")

if __name__=="__main__":
    main()
