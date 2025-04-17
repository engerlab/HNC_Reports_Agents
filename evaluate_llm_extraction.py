#!/usr/bin/env python3
# evaluate_llm_extractions.py

"""
This script evaluates the LLM extractions from the clinician evaluation forms.
1) Finds a 'Time to complete' column by name instead of assuming the last column is time.
2) Fixes the TP/FP/TN/FN logic to match your definitions:
     - If LLM predicted positive (value != 'Not inferred') and majority is 'Agree' => TP
     - If LLM predicted positive and majority is 'Disagree' => FP
     - If LLM predicted negative (value == 'Not inferred') and majority is 'Agree' => TN
     - If LLM predicted negative and majority is 'Disagree' => FN
3) Uses statsmodels' fleiss_kappa for more accurate values.
4) Renames `llm_ratings.csv` => `llm_ratings_raw.csv`, 
   plus a new `llm_ratings_summary.csv` with average & SD of each reviewer’s ratings.
5) Adds an overall confusion matrix CSV: `confusion_matrix_overall.csv`
   in addition to the per-field matrix.

`pip install statsmodels` to ensure fleiss_kappa is available.
"""

import argparse
import os
import re
import math

import pandas as pd
import numpy as np

# for Fleiss' kappa
import statsmodels.stats.inter_rater as ir

from data_sources import DATA_SOURCES

###############################################################################
# CONFIGURATION OF EXPECTED FIELDS
###############################################################################
EXPECTED_FIELDS = [
    "Sex",
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
    "Smoking_History",
    "Alcohol_Consumption",
    "Pack_Years",
    "Patient_Symptoms_at_Presentation",
    "Treatment_Recommendations",
    "Follow_Up_Plans",
    "HPV_Status",
    "Patient_History_Status_Prior_Conditions",
    "Patient_History_Status_Previous_Treatments",
    "Clinical_Assessments_Radiological_Lesions",
    "Clinical_Assessments_SUV_from_PET_scans",
    "Charlson_Comorbidity_Score",
    "Karnofsky_Performance_Status",
    "ECOG_Performance_Status",
]

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def extract_field_from_colname(colname: str) -> str:
    """
    Attempt to parse the bracket [XYZ: ABC] portion to see if 'XYZ' is in EXPECTED_FIELDS.
    e.g. "Please indicate if you agree... [Sex: F]" -> "Sex".
    Returns the field name if matched, else None.
    """
    match = re.search(r"\[(.*?)\]", colname)
    if not match:
        return None
    bracket_content = match.group(1)  # e.g. "Sex: F"
    parts = bracket_content.split(":")
    if len(parts) < 2:
        return None
    candidate_field = parts[0].strip()
    if candidate_field in EXPECTED_FIELDS:
        return candidate_field
    return None

def extract_llm_value(colname: str) -> str:
    """
    Return the portion after the colon in a bracket, e.g. "[Sex: F]" => "F".
    So we can see if it's "Not inferred" or an actual value.
    """
    match = re.search(r"\[(.*?)\]", colname)
    if not match:
        return None
    bracket_content = match.group(1)
    parts = bracket_content.split(":")
    if len(parts) < 2:
        return None
    value_part = parts[1].strip()
    return value_part

def find_time_column(df: pd.DataFrame) -> str:
    """
    Attempt to locate a column that likely represents 'Time to complete'.
    We'll look for any column whose name includes 'time' or 'Time' or 'complete'.
    If you know the exact column name, you can do an exact match.
    If no match is found, returns None.
    """
    possible_time_cols = []
    for col in df.columns:
        low = col.lower()
        if "time" in low and "complete" in low:
            possible_time_cols.append(col)

    if len(possible_time_cols) == 1:
        return possible_time_cols[0]
    elif len(possible_time_cols) > 1:
        # If multiple columns look like time columns, pick the first
        # or raise an error. For now we pick the first.
        return possible_time_cols[0]
    else:
        return None

def parse_single_form(df: pd.DataFrame, form_name: str) -> pd.DataFrame:
    """
    Convert the wide DataFrame (one row per reviewer submission) into a "long" format:
    columns: 
      [FormName, ReviewerEmail, CaseIndex, FieldName, LLM_Value, ReviewerResponse,
       Comment, Rating, TimeToComplete (float or str if not convertible)]

    We search for a 'Time to complete' column by name,
    then we assume the rest of the columns are sets of 32 columns per case
    (30 fields + 1 comment + 1 rating).
    """

    # A) Identify the time column if any
    time_col = find_time_column(df)
    if not time_col:
        print(f"[WARNING] Could not find a 'time to complete' column in form: {form_name}.")
        print("The script will leave TimeToComplete = None for these rows.")
        # We can proceed without time
    else:
        # Attempt to convert that column to float if possible
        # If not convertible, we'll just store it as string
        try:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        except Exception:
            pass  # fallback, they remain object type

    # B) We assume:
    #    col 0 = Timestamp
    #    col 1 = Email
    #    The 'time to complete' col is separate
    #    The rest must be chunked in sets of 32 columns per case
    # We'll gather relevant columns for data chunking (all except 0,1, time_col)
    skip_cols = set()
    skip_cols.update(df.columns[:2])  # skip first 2
    if time_col:
        skip_cols.add(time_col)

    data_cols = [c for c in df.columns if c not in skip_cols]

    long_records = []

    for row_i in range(len(df)):
        row_data = df.iloc[row_i]
        reviewer_email = row_data.iloc[1]  # col=1 => Email
        # Grab the time value if it exists
        time_val = None
        if time_col:
            time_val = row_data[time_col]

        # chunk data_cols in sets of 32
        # if you consistently have 5 or 6 cases, do a while loop to cut slices of 32
        # if data_cols is not exactly multiple of 32, we adapt
        chunk_size = 32
        n_data_cols = len(data_cols)
        # number of cases = n_data_cols // chunk_size or partial
        # if forms differ in #cases, they'll yield partial last chunk. We'll skip partial chunk if needed.
        n_cases = n_data_cols // chunk_size
        # if there's leftover columns, we ignore them or warn
        leftover = n_data_cols % chunk_size
        if leftover != 0:
            print(f"[INFO] {form_name} has leftover columns (not multiple of 32). We will parse the first {n_cases} cases only.")
        idx = 0
        for case_i in range(1, n_cases+1):
            chunk_cols = data_cols[idx: idx+chunk_size]
            idx += chunk_size

            # The first 30 columns in chunk => fields
            # the 31st => comment
            # the 32nd => rating
            fields_part = chunk_cols[:30]
            comment_col = chunk_cols[30] if len(chunk_cols) > 30 else None
            rating_col = chunk_cols[31] if len(chunk_cols) > 31 else None

            comment_val = row_data[comment_col] if comment_col else None
            rating_val = row_data[rating_col] if rating_col else None

            for c_field_col in fields_part:
                colname = c_field_col
                response = row_data[c_field_col]  # "Agree"/"Disagree" or NaN
                field_name = extract_field_from_colname(colname)
                llm_value = extract_llm_value(colname)
                if not field_name:
                    # skip columns that don't parse
                    continue

                record = {
                    "FormName": form_name,
                    "ReviewerEmail": reviewer_email,
                    "CaseIndex": case_i,
                    "FieldName": field_name,
                    "LLM_Value": llm_value,
                    "ReviewerResponse": str(response).strip() if pd.notna(response) else "",
                    "Comment": comment_val,
                    "Rating": rating_val,
                    "TimeToComplete": time_val,
                }
                long_records.append(record)

    return pd.DataFrame(long_records)

def fleiss_kappa_agree_disagree(df_field: pd.DataFrame):
    """
    Compute Fleiss' kappa using statsmodels, for 3 raters, 2 categories (Agree, Disagree).
    1) For each case, count #Agree, #Disagree among the 3 raters => row = [disagree_count, agree_count]
    2) Use statsmodels' fleiss_kappa(...) on that Nx2 array.
    """
    # Convert "Agree" -> 1, anything else -> 0
    df_field["IsAgree"] = df_field["ReviewerResponse"].apply(lambda x: 1 if x.lower()=="agree" else 0)

    # group by case => sum up the 3 responses
    grouped = df_field.groupby("CaseIndex")["IsAgree"].agg(["sum","count"]).reset_index()
    # sum = how many "Agree"
    # count = number of raters (should be 3)
    # disagrees = count - sum

    # Build Nx2 array
    # row i = [disagrees, agrees]
    data_rows = []
    for _, row in grouped.iterrows():
        agrees = int(row["sum"])
        total = int(row["count"])
        disagrees = total - agrees
        data_rows.append([disagrees, agrees])
    arr = np.array(data_rows)

    if len(arr) < 1:
        return np.nan

    # statsmodels requires aggregator
    # But for 2 categories, we can do direct:
    # table, cats = ir.aggregate_raters(arr)
    # kappa = ir.fleiss_kappa(table)
    # or we can pass arr directly to aggregate_raters
    table, cats = ir.aggregate_raters(arr)
    kappa_val = ir.fleiss_kappa(table)
    return kappa_val


def interpret_llm_prediction(llm_value: str) -> bool:
    """
    Return True if LLM predicted a positive (some value),
    False if LLM predicted 'Not inferred' or blank.
    """
    if not isinstance(llm_value, str):
        return False
    return llm_value.strip().lower() != "not inferred"


def get_tp_fp_tn_fn(llm_pred_positive: bool, majority_says_correct: bool):
    """
    Updated logic matching your definitions:
      - If LLM_pred_positive & majority_says_correct => TP
      - If LLM_pred_positive & majority_says_incorrect => FP
      - If LLM_pred_negative & majority_says_correct => TN
      - If LLM_pred_negative & majority_says_incorrect => FN
    """
    if llm_pred_positive and majority_says_correct:
        return (1,0,0,0)  # TP
    elif llm_pred_positive and not majority_says_correct:
        return (0,1,0,0)  # FP
    elif not llm_pred_positive and majority_says_correct:
        return (0,0,1,0)  # TN
    else:
        return (0,0,0,1)  # FN

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Extractions (Updated)")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to store output CSV files (default: current directory)."
    )
    args = parser.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # (A) Download & parse all forms
    all_parsed = []
    for form_name, csv_url in DATA_SOURCES.items():
        print(f"[INFO] Loading form {form_name} from {csv_url}")
        df_raw = pd.read_csv(csv_url)
        parsed_df = parse_single_form(df_raw, form_name)
        all_parsed.append(parsed_df)

    master_df = pd.concat(all_parsed, ignore_index=True)

    # Save the "master_long_data.csv"
    master_long_path = os.path.join(outdir, "master_long_data.csv")
    master_df.to_csv(master_long_path, index=False)
    print(f"=> Saved combined long data to: {master_long_path}")

    # (B) Fleiss' Kappa by field using statsmodels
    fleiss_rows = []
    for fld in EXPECTED_FIELDS:
        df_fld = master_df[master_df["FieldName"] == fld].copy()
        if len(df_fld) == 0:
            continue
        kappa_val = fleiss_kappa_agree_disagree(df_fld)
        fleiss_rows.append({"FieldName": fld, "FleissKappa": kappa_val})
    df_kappa = pd.DataFrame(fleiss_rows).sort_values("FieldName").reset_index(drop=True)

    fleiss_kappa_csv = os.path.join(outdir, "fleiss_kappa_by_field.csv")
    df_kappa.to_csv(fleiss_kappa_csv, index=False)
    print(f"=> Saved Fleiss' kappa results to {fleiss_kappa_csv}")

    # (C) Confusion matrix: compute majority correctness
    # We group by (FormName, CaseIndex, FieldName) => 3 rows, see how many "Agree" vs "Disagree"
    grouped = master_df.groupby(["FormName", "CaseIndex", "FieldName"], as_index=False)
    confusion_records = []
    for (fm, cs, fld), subdf in grouped:
        # Among the 3 reviewers, how many said "Agree"?
        num_agree = (subdf["ReviewerResponse"].str.lower() == "agree").sum()
        # If >=2 => majority says correct
        majority_correct = (num_agree >= 2)

        # LLM predicted positive if any of the row's LLM_Value != "Not inferred" 
        # but in practice it should be the same for all 3. We'll just take the first:
        llm_value = subdf["LLM_Value"].iloc[0]
        pred_positive = interpret_llm_prediction(llm_value)

        tp, fp, tn, fn = get_tp_fp_tn_fn(pred_positive, majority_correct)
        confusion_records.append({
            "FormName": fm,
            "CaseIndex": cs,
            "FieldName": fld,
            "LLM_Value": llm_value,
            "MajoritySaysCorrect": majority_correct,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
    df_cm = pd.DataFrame(confusion_records)

    # Save confusion_matrix_raw
    cm_raw_path = os.path.join(outdir, "confusion_matrix_raw.csv")
    df_cm.to_csv(cm_raw_path, index=False)
    print(f"=> Saved confusion_matrix_raw to {cm_raw_path}")

    # Summarize by field
    summary_cm = df_cm.groupby("FieldName")[["TP","FP","TN","FN"]].sum().reset_index()

    # add metrics
    def calc_metrics(row):
        TP = row["TP"]; FP = row["FP"]; TN = row["TN"]; FN = row["FN"]
        total = TP+FP+TN+FN
        acc = (TP + TN)/total if total>0 else np.nan
        prec = TP/(TP+FP) if (TP+FP)>0 else np.nan
        rec = TP/(TP+FN) if (TP+FN)>0 else np.nan
        f1 = (2*prec*rec)/(prec+rec) if prec is not np.nan and rec is not np.nan and (prec+rec)>0 else np.nan
        return pd.Series({
            "TP":TP, "FP":FP, "TN":TN, "FN":FN,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

    df_field_metrics = summary_cm.apply(calc_metrics, axis=1)
    metrics_by_field = pd.concat([summary_cm["FieldName"], df_field_metrics], axis=1)
    field_metrics_path = os.path.join(outdir, "extraction_metrics_by_field.csv")
    metrics_by_field.to_csv(field_metrics_path, index=False)
    print(f"=> Saved per-field extraction metrics to {field_metrics_path}")

    # Also produce an overall confusion matrix across ALL fields
    overall_vals = df_cm[["TP","FP","TN","FN"]].sum()
    TP = overall_vals["TP"]
    FP = overall_vals["FP"]
    TN = overall_vals["TN"]
    FN = overall_vals["FN"]
    total = TP+FP+TN+FN
    acc = (TP+TN)/total if total>0 else np.nan
    prec = TP/(TP+FP) if (TP+FP)>0 else np.nan
    rec = TP/(TP+FN) if (TP+FN)>0 else np.nan
    f1 = (2*prec*rec)/(prec+rec) if prec is not np.nan and rec is not np.nan and (prec+rec)>0 else np.nan

    df_overall = pd.DataFrame([{
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
    }])
    overall_cm_path = os.path.join(outdir, "confusion_matrix_overall.csv")
    df_overall.to_csv(overall_cm_path, index=False)
    print(f"=> Saved overall confusion matrix to {overall_cm_path}")

    # (D) LLM ratings
    # rename the raw to "llm_ratings_raw.csv"
    # but also produce a summary: average & SD rating per reviewer

    # first group by (FormName, CaseIndex, ReviewerEmail) to get unique rating
    # actually we can just do group by the same, or simpler approach:
    df_ratings = master_df[["FormName","CaseIndex","ReviewerEmail","Rating"]].drop_duplicates()
    # rename the file
    rating_raw_path = os.path.join(outdir, "llm_ratings_raw.csv")
    df_ratings.to_csv(rating_raw_path, index=False)
    print(f"=> Saved raw LLM ratings to {rating_raw_path}")

    # Summarize: average rating for each reviewer across all forms & cases
    # ignoring non-numeric or missing ratings
    df_ratings["NumericRating"] = pd.to_numeric(df_ratings["Rating"], errors="coerce")
    reviewer_stats = df_ratings.groupby("ReviewerEmail")["NumericRating"].agg(["count","mean","std"])
    reviewer_stats = reviewer_stats.reset_index()
    # rename columns
    reviewer_stats.rename(columns={
        "count":"N_cases",
        "mean":"MeanRating",
        "std":"SDRating"
    }, inplace=True)

    rating_summary_path = os.path.join(outdir, "llm_ratings_summary.csv")
    reviewer_stats.to_csv(rating_summary_path, index=False)
    print(f"=> Saved LLM rating summary to {rating_summary_path}")

    # (E) Reviewer times
    # In parse_single_form, we store "TimeToComplete" from the recognized column, if found
    df_time = master_df[["FormName","ReviewerEmail","CaseIndex","TimeToComplete"]].drop_duplicates()
    # One row per (Form, Reviewer, Case)
    # But the time is at the form level, so it repeats for each case => we group by (Form, Reviewer)
    # then take the unique time
    time_groups = df_time.groupby(["FormName","ReviewerEmail"], as_index=False)
    recs = []
    for (fm,rv), subdf in time_groups:
        tvals = subdf["TimeToComplete"].dropna().unique()
        if len(tvals)==1:
            tval = tvals[0]
        elif len(tvals)>1:
            # user reported multiple times? we pick the first or average?
            tval = tvals[0]
            print(f"[WARNING] multiple distinct times found for {fm}-{rv}: {tvals}. Using first={tval}")
        else:
            tval = None
        n_cases = subdf["CaseIndex"].nunique()
        recs.append({
            "FormName": fm,
            "ReviewerEmail": rv,
            "TimeMinutesForWholeForm": tval,
            "NumCasesInForm": n_cases
        })
    df_times = pd.DataFrame(recs)

    def compute_avg_time(row):
        if pd.isna(row["TimeMinutesForWholeForm"]) or row["NumCasesInForm"]==0:
            return None
        return float(row["TimeMinutesForWholeForm"])/row["NumCasesInForm"]

    df_times["AvgTimePerCase"] = df_times.apply(compute_avg_time, axis=1)
    times_raw_path = os.path.join(outdir, "reviewer_times_raw.csv")
    df_times.to_csv(times_raw_path, index=False)
    print(f"=> Saved reviewer_times_raw to {times_raw_path}")

    # overall average time per case
    overall_times = df_times.dropna(subset=["AvgTimePerCase"])["AvgTimePerCase"].values
    if len(overall_times) > 0:
        mean_time = np.mean(overall_times)
        sd_time = np.std(overall_times, ddof=1)
        n = len(overall_times)
        sem = sd_time/math.sqrt(n) if n>1 else 0
        z_95 = 1.96
        lower = mean_time - z_95*sem
        upper = mean_time + z_95*sem
        df_time_summary = pd.DataFrame([{
            "MeanTimePerCase": mean_time,
            "SD": sd_time,
            "N": n,
            "CI_95_Lower": lower,
            "CI_95_Upper": upper
        }])
        time_summary_path = os.path.join(outdir, "time_per_case_summary.csv")
        df_time_summary.to_csv(time_summary_path, index=False)
        print(f"=> Overall average time ~ {mean_time:.2f} mins (95% CI: {lower:.2f}-{upper:.2f}), n={n}")
        print(f"=> Saved time_per_case_summary to {time_summary_path}")
    else:
        print("[INFO] No valid numeric 'TimeToComplete' data found. Skipped summary stats.")

    print("Analysis complete!")

if __name__ == "__main__":
    main()



# Usage 
# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/evaluate_llm_extraction.py --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Clinician_Evaluation/
