#!/usr/bin/env python3
"""
extract_fields_to_csv.py

- p16_Status, HPV_Status
- Smoking_History
- Pack_Years
- Alcohol_Consumption => only "Drinker" or "Non-Drinker" or blank if not found or "Not inferred".
- Charlson_Comorbidity_Score => integer or blank
- Karnofsky_Performance_Status => integer or blank
- ECOG_Performance_Status => integer or blank

Example usage:

  python extract_fields_to_csv.py \
    --input_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
    --id_list "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/total_unique_ids.csv" \
    --out_path_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results"
"""

import os
import math
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

##############################################################################
# Helpers
##############################################################################

def parse_line_for_field(summary_file, field_name):
    """
    Return the substring after "FieldName:" from the summary file line,
    or None if not found.
    """
    if not os.path.isfile(summary_file):
        return None

    prefix = field_name + ":"
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith(prefix):
                val_str = line_stripped[len(prefix):].strip()
                return val_str
    return None

def parse_numeric(val_str):
    """
    For lines like "3 [stuff]" or just "3" or "5", we extract
    only the integer part. If no integer can be found, return pd.NA.

    e.g. "3 [stuff]" => "3"
    If parse fails => pd.NA
    """
    if val_str is None:
        return pd.NA

    parts = val_str.split()
    if not parts:
        return pd.NA

    first_chunk = parts[0]
    try:
        return int(first_chunk)
    except ValueError:
        return pd.NA

def simplify_alcohol(raw_val):
    """
    Return "Drinker" or "Non-Drinker" or pd.NA (to become blank).
    If raw_val is None => pd.NA
    Convert to lowercase:
      if "not inferred" in text => pd.NA
      elif "non" in text => "Non-Drinker"
      else => "Drinker"
    """
    if raw_val is None:
        return pd.NA

    text = raw_val.lower()
    if "not inferred" in text:
        return pd.NA
    elif "non" in text:
        # Any occurrence of "non" means we store "Non-Drinker"
        return "Non-Drinker"
    else:
        # If we found the line at all, assume "Drinker"
        return "Drinker"

##############################################################################
# Main
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Base directory, containing text_summaries/path_consult_reports/<pid>/path_consult_reports_summary.txt")
    parser.add_argument("--id_list", required=True,
                        help="CSV with column 'patient_id' listing all unique IDs.")
    parser.add_argument("--out_path_dir", required=True,
                        help="Output directory for Path_Structured/ and Cons_Structured/ subfolders + CSVs.")
    args = parser.parse_args()

    # Load IDs
    df_ids = pd.read_csv(args.id_list)
    patient_ids = df_ids["patient_id"].astype(str).tolist()
    logger.info(f"Loaded {len(patient_ids)} patient IDs from {args.id_list}")

    # Create subdirs
    path_subdir = os.path.join(args.out_path_dir, "Path_Structured")
    cons_subdir = os.path.join(args.out_path_dir, "Cons_Structured")
    os.makedirs(path_subdir, exist_ok=True)
    os.makedirs(cons_subdir, exist_ok=True)

    def summary_path(pid):
        return os.path.join(
            args.input_dir,
            "text_summaries",
            "path_consult_reports",
            pid,
            "path_consult_reports_summary.txt"
        )

    # We'll build lists of dicts
    path_records = []
    cons_records = []

    for pid in patient_ids:
        spath = summary_path(pid)

        # ========== Path fields ==============
        val_p16 = parse_line_for_field(spath, "p16_Status")
        val_hpv = parse_line_for_field(spath, "HPV_Status")

        if val_p16 is None:
            val_p16 = "Not inferred"
        if val_hpv is None:
            val_hpv = "Not inferred"

        path_records.append({
            "patient_id": pid,
            "p16_Status": val_p16,
            "HPV_Status": val_hpv,
        })

        # ========== Cons fields ==============
        val_smoke  = parse_line_for_field(spath, "Smoking_History")
        val_pack   = parse_line_for_field(spath, "Pack_Years")
        val_alcraw = parse_line_for_field(spath, "Alcohol_Consumption")
        val_char   = parse_line_for_field(spath, "Charlson_Comorbidity_Score")
        val_kps    = parse_line_for_field(spath, "Karnofsky_Performance_Status")
        val_ecog   = parse_line_for_field(spath, "ECOG_Performance_Status")

        # For Alcohol, store Drinker / Non-Drinker / blank
        alc_clean = simplify_alcohol(val_alcraw)

        # For Smoking, if missing => "Not inferred"
        if val_smoke is None:
            val_smoke = "Not inferred"
        # For pack yrs, parse numeric?
        # If user wants integer => parse
        # else keep as string
        # We'll parse it:
        def parse_pack_yrs(s):
            if not s:
                return pd.NA
            s_lower = s.lower()
            if "not inferred" in s_lower:
                return pd.NA
            # else try to parse int
            try:
                return int(s)
            except:
                return pd.NA

        pack_num = parse_pack_yrs(val_pack)

        # For numeric parse => char, kps, ecog
        char_num = parse_numeric(val_char)
        kps_num  = parse_numeric(val_kps)
        ecog_num = parse_numeric(val_ecog)

        cons_records.append({
            "patient_id": pid,
            "Smoking_History": val_smoke,
            "Pack_Years": pack_num,  # We'll store as Int64 below
            "Alcohol_Consumption": alc_clean,  # "Drinker", "Non-Drinker", or pd.NA
            "Charlson_Comorbidity_Score": char_num,
            "Karnofsky_Performance_Status": kps_num,
            "ECOG_Performance_Status": ecog_num
        })

    # ========== Build Path DF ==============
    df_path = pd.DataFrame(path_records)

    # Sort p16 => Positive, Negative, else
    def sort_p16(val):
        v = val.lower()
        if "positive" in v:
            return 0
        elif "negative" in v:
            return 1
        return 2

    df_path["sort_key"] = df_path["p16_Status"].apply(sort_p16)
    df_path.sort_values(by="sort_key", inplace=True)
    df_path.drop(columns=["sort_key"], inplace=True)

    out_csv_path = os.path.join(path_subdir, "p16_hpv.csv")
    df_path.to_csv(out_csv_path, index=False)
    logger.info(f"Saved {len(df_path)} => {out_csv_path}")

    # ========== Build Cons DF ==============
    df_cons = pd.DataFrame(cons_records)
    df_cons.sort_values(by="patient_id", inplace=True)

    # Convert numeric columns to Int64 => then replace <NA> => blank
    numeric_cols = [
        "Pack_Years",
        "Charlson_Comorbidity_Score",
        "Karnofsky_Performance_Status",
        "ECOG_Performance_Status"
    ]
    for col in numeric_cols:
        df_cons[col] = df_cons[col].astype("Int64")

    # Alcohol_Consumption => if pd.NA => becomes blank
    # And same for numeric if <NA>
    # We'll do the trick of converting to string => replacing <NA> => ""
    for col in df_cons.columns:
        if pd.api.types.is_sparse(df_cons[col]) or pd.api.types.is_string_dtype(df_cons[col]):
            # might not matter
            pass

    # We'll forcibly convert all columns to string, except we do it carefully for the numeric cols
    # simpler approach: for numeric, we do => df_cons[col].astype(str).replace("<NA>","")
    # for the others that might be pd.NA => also do the same
    for col in df_cons.columns:
        df_cons[col] = df_cons[col].astype(str).replace("<NA>", "")

    cons_csv = os.path.join(cons_subdir, "cons_fields.csv")
    df_cons.to_csv(cons_csv, index=False)
    logger.info(f"Saved {len(df_cons)} => {cons_csv}")

if __name__=="__main__":
    main()

# usage 
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/extract_fields_to_csv.py \
#   --input_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13" \
#   --id_list "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/total_unique_ids.csv" \
#   --out_path_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results"
