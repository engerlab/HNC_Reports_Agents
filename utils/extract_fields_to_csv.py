#!/usr/bin/env python3
"""
extract_fields_to_csv.py (UPDATED for Exp14 with extended fields)

This script reads a set of patient IDs, then for each patient:
    1) Finds the final summary file at:
       <input_dir>/text_summaries/mode2_combined_twostep/<pid>/path_consult_reports_summary.txt
    2) Parses out various "Path" fields into path_fields.csv
    3) Parses out various "Cons" fields into cons_fields.csv
    4) Writes CSVs under <out_path_dir>/Path_Structured/ and <out_path_dir>/Cons_Structured/

Path fields (path_fields.csv):
  - Anatomic_Site_of_Lesion
  - Pathological_TNM
  - Clinical_TNM
  - Primary_Tumor_Size
  - Tumor_Type_Differentiation
  - Pathology_Details
  - Lymph_Node_Status_Presence_Absence
  - Lymph_Node_Status_Number_of_Positve_Lymph_Nodes
  - Lymph_Node_Status_Extranodal_Extension
  - Resection_Margins
  - p16_Status
  - Immunohistochemical_profile
  - EBER_Status
  - Lymphovascular_Invasion_Status
  - Perineural_Invasion_Status
  - HPV_Status

Cons fields (cons_fields.csv):
  - Sex
  - Smoking_History
  - Alcohol_Consumption
  - Pack_Years
  - Patient_Symptoms_at_Presentation
  - Treatment_Recommendations
  - Follow_Up_Plans
  - Patient_History_Status_Prior_Conditions
  - Patient_History_Status_Previous_Treatments
  - Clinical_Assessments_Radiological_Lesions
  - Clinical_Assessments_SUV_from_PET_scans
  - Charlson_Comorbidity_Score
  - Karnofsky_Performance_Status
  - ECOG_Performance_Status

Example usage:
  python extract_fields_to_csv.py \
    --input_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14" \
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
# Utility functions
##############################################################################

def parse_line_for_field(summary_file, field_name):
    """
    Looks in 'summary_file' for a line that starts with "{field_name}:"
    and returns the remainder of that line as a string, or None if:
      - The file doesn't exist
      - There's no line matching 'field_name:'
    """
    if not os.path.isfile(summary_file):
        return None

    prefix = field_name + ":"
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith(prefix):
                return line_stripped[len(prefix):].strip()

    return None

def parse_numeric(val_str):
    """
    If val_str looks like "3" or "3 something", returns int(3).
    If parsing fails or val_str is None, returns pd.NA.
    """
    if val_str is None:
        return pd.NA
    parts = val_str.split()
    if not parts:
        return pd.NA
    try:
        return int(parts[0])
    except ValueError:
        return pd.NA

def parse_pack_years(raw_val):
    """
    Convert pack years to an integer if possible.
    If raw_val includes "not inferred" or fails to parse => pd.NA.
    """
    if not raw_val:
        return pd.NA
    lower_val = raw_val.lower()
    if "not inferred" in lower_val:
        return pd.NA
    try:
        return int(raw_val)
    except:
        return pd.NA

def simplify_alcohol(raw_val):
    """
    Return "Drinker" / "Non-Drinker" / pd.NA based on 'raw_val'.
    If raw_val is None => pd.NA
    - if "not inferred" in text => pd.NA
    - elif "non" => "Non-Drinker"
    - else => "Drinker"
    """
    if raw_val is None:
        return pd.NA

    text = raw_val.lower()
    if "not inferred" in text:
        return pd.NA
    elif "non" in text:
        return "Non-Drinker"
    else:
        return "Drinker"


##############################################################################
# Main
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Base directory with text_summaries/mode2_combined_twostep/<pid>/path_consult_reports_summary.txt")
    parser.add_argument("--id_list", required=True,
                        help="CSV with column 'patient_id' for all unique IDs to parse")
    parser.add_argument("--out_path_dir", required=True,
                        help="Output directory for Path_Structured/ and Cons_Structured/ subfolders + CSVs.")
    args = parser.parse_args()

    # 1) Read patient IDs
    df_ids = pd.read_csv(args.id_list)
    patient_ids = df_ids["patient_id"].astype(str).tolist()
    logger.info(f"Loaded {len(patient_ids)} patient IDs from {args.id_list}")

    # 2) Make output subfolders
    path_subdir = os.path.join(args.out_path_dir, "Path_Structured")
    cons_subdir = os.path.join(args.out_path_dir, "Cons_Structured")
    os.makedirs(path_subdir, exist_ok=True)
    os.makedirs(cons_subdir, exist_ok=True)

    # 3) Summaries are at: input_dir/text_summaries/mode2_combined_twostep/<pid>/path_consult_reports_summary.txt
    def summary_path(pid):
        return os.path.join(
            args.input_dir,
            "text_summaries",
            "mode2_combined_twostep",
            pid,
            "path_consult_reports_summary.txt"
        )

    # 4) Fields we want
    path_fields = [
        "Anatomic_Site_of_Lesion",
        "Pathological_TNM",
        "Clinical_TNM",
        "Primary_Tumor_Size",
        "Tumor_Type_Differentiation",
        "Pathology_Details",
        "Lymph_Node_Status_Presence_Absence",
        "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes",
        "Lymph_Node_Status_Extranodal_Extension",
        "Resection_Margins",
        "p16_Status",
        "Immunohistochemical_profile",
        "EBER_Status",
        "Lymphovascular_Invasion_Status",
        "Perineural_Invasion_Status",
        "HPV_Status"
    ]

    cons_fields = [
        "Sex",
        "Smoking_History",
        "Alcohol_Consumption",
        "Pack_Years",
        "Patient_Symptoms_at_Presentation",
        "Treatment_Recommendations",
        "Follow_Up_Plans",
        "Patient_History_Status_Prior_Conditions",
        "Patient_History_Status_Previous_Treatments",
        "Clinical_Assessments_Radiological_Lesions",
        "Clinical_Assessments_SUV_from_PET_scans",
        "Charlson_Comorbidity_Score",
        "Karnofsky_Performance_Status",
        "ECOG_Performance_Status"
    ]

    # 5) We'll build two lists of dicts
    path_records = []
    cons_records = []

    for pid in patient_ids:
        spath = summary_path(pid)

        # A) PATH FIELDS
        path_dict = {"patient_id": pid}
        for fld in path_fields:
            val = parse_line_for_field(spath, fld)
            if val is None:
                val = "Not inferred"  # or keep as None => blank after?
            path_dict[fld] = val
        path_records.append(path_dict)

        # B) CONS FIELDS
        cons_dict = {"patient_id": pid}
        for fld in cons_fields:
            raw_val = parse_line_for_field(spath, fld)
            if raw_val is None:
                # For numeric or special parse, we'll handle below
                raw_val = None

            # If the field is straightforward (just store text):
            # We'll fill in special logic for some fields:
            if fld == "Alcohol_Consumption":
                cons_dict[fld] = simplify_alcohol(raw_val)
            elif fld == "Pack_Years":
                cons_dict[fld] = parse_pack_years(raw_val)
            elif fld in ("Charlson_Comorbidity_Score",
                         "Karnofsky_Performance_Status",
                         "ECOG_Performance_Status"):
                cons_dict[fld] = parse_numeric(raw_val)
            elif fld == "Smoking_History":
                if raw_val is None:
                    cons_dict[fld] = "Not inferred"
                else:
                    cons_dict[fld] = raw_val
            else:
                # Default handling => raw_val or "Not inferred"
                if raw_val is None:
                    cons_dict[fld] = "Not inferred"
                else:
                    cons_dict[fld] = raw_val

        cons_records.append(cons_dict)

    # 6) Build Path DF, fix any final formatting
    df_path = pd.DataFrame(path_records)
    # Example: if you prefer to keep "Not inferred" as-is, that's fine
    # Otherwise, you can replace "Not inferred" => "" if you want them blank

    path_csv = os.path.join(path_subdir, "path_fields.csv")
    df_path.to_csv(path_csv, index=False)
    logger.info(f"Wrote PATH fields to => {path_csv}, rows={len(df_path)}")

    # 7) Build Cons DF, handle numeric columns + final formatting
    df_cons = pd.DataFrame(cons_records)

    # Some columns should be numeric => store them as Int64, then convert <NA> to blank
    numeric_cols = [
        "Pack_Years",
        "Charlson_Comorbidity_Score",
        "Karnofsky_Performance_Status",
        "ECOG_Performance_Status"
    ]
    for col in numeric_cols:
        if col in df_cons.columns:
            df_cons[col] = df_cons[col].astype("Int64")

    # Convert everything to string, replace <NA> => ""
    for col in df_cons.columns:
        df_cons[col] = df_cons[col].astype(str).replace("<NA>", "")

    cons_csv = os.path.join(cons_subdir, "cons_fields.csv")
    df_cons.to_csv(cons_csv, index=False)
    logger.info(f"Wrote CONS fields to => {cons_csv}, rows={len(df_cons)}")


if __name__ == "__main__":
    main()


# usage 
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/utils/extract_fields_to_csv.py \
#   --input_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14" \
#   --id_list "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/patient_stats/total_unique_ids.csv" \
#   --out_path_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results"
