#!/usr/bin/env python3
# evaluate_llm_extractions.py

import argparse
import os
import re
import math

import pandas as pd
import numpy as np
from scipy import stats  # for confidence intervals if you want

# 1) Import the dictionary of CSV export links
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
    Attempt to parse the bracket [XYZ: ABC] portion to see if 'XYZ' is in our EXPECTED_FIELDS.
    e.g. "Please indicate if you agree... [Sex: F]" -> "Sex"
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
    Return the portion after the colon in a bracket, e.g. "Sex: F" => "F".
    So we can see if it's "Not inferred" or something else.
    """
    match = re.search(r"\[(.*?)\]", colname)
    if not match:
        return None
    bracket_content = match.group(1)  # e.g. "Sex: F"
    parts = bracket_content.split(":")
    if len(parts) < 2:
        return None
    value_part = parts[1].strip()
    return value_part  # e.g. "F", "Not inferred", etc.

def parse_single_form(df: pd.DataFrame, form_name: str) -> pd.DataFrame:
    """
    Convert the wide DataFrame (one row per reviewer submission) into a "long" format:
    columns: [FormName, ReviewerEmail, CaseIndex, FieldName, LLM_Value, ReviewerResponse,
              Comment, Rating, TimeToComplete].
    
    We assume:
      col 0 = Timestamp,
      col 1 = Email,
      then sets of 32 columns per case (30 fields + 1 comment + 1 rating),
      final col = "TimeToComplete" for the entire form.
    
    Adjust logic if your forms differ.
    """
    n_cols = len(df.columns)
    time_col_idx = n_cols - 1  # last column is "time to complete"

    long_records = []

    for row_i in range(len(df)):
        row_data = df.iloc[row_i]
        reviewer_email = row_data.iloc[1]  # col=1 => email address
        time_to_complete = row_data.iloc[time_col_idx]

        data_cols_end = time_col_idx - 1
        num_data_cols = data_cols_end - 2 + 1  # from col index 2..(time_col_idx-1)
        if num_data_cols % 32 != 0:
            print(f"[WARNING] {form_name} row {row_i}: # data cols not multiple of 32. Check parsing logic.")
        n_cases = num_data_cols // 32

        col_start = 2
        for case_idx in range(1, n_cases+1):
            # columns for this case => col_start..col_start+31
            for c_field_idx in range(col_start, col_start+30):
                colname = df.columns[c_field_idx]
                response = row_data.iloc[c_field_idx]  # "Agree"/"Disagree"

                field_name = extract_field_from_colname(colname)
                llm_value = extract_llm_value(colname)

                if not field_name:
                    continue

                record = {
                    "FormName": form_name,
                    "ReviewerEmail": reviewer_email,
                    "CaseIndex": case_idx,
                    "FieldName": field_name,
                    "LLM_Value": llm_value,
                    "ReviewerResponse": response,
                    "Comment": None,
                    "Rating": None,
                    "TimeToComplete": time_to_complete,
                }
                long_records.append(record)

            # comment col
            comment_col_idx = col_start + 30
            comment_val = row_data.iloc[comment_col_idx] if comment_col_idx <= data_cols_end else None
            # rating col
            rating_col_idx = col_start + 31
            rating_val = row_data.iloc[rating_col_idx] if rating_col_idx <= data_cols_end else None

            # fill in the comment & rating for all rows from this case
            for r in long_records:
                if (r["FormName"] == form_name 
                    and r["ReviewerEmail"] == reviewer_email
                    and r["CaseIndex"] == case_idx):
                    r["Comment"] = comment_val
                    r["Rating"] = rating_val

            col_start += 32

    return pd.DataFrame(long_records)

def compute_fleiss_kappa_for_field(df_field: pd.DataFrame) -> float:
    """
    Compute Fleiss' Kappa for 3 raters, 2 categories (Agree/Disagree).
    """
    df_field["AgreeBinary"] = df_field["ReviewerResponse"].apply(
        lambda x: 1 if str(x).strip().lower() == "agree" else 0
    )

    grouped = df_field.groupby("CaseIndex")["AgreeBinary"].agg(["sum","count"])
    # sum = # of agrees, count=3 (3 reviewers)
    # 2 categories => agrees vs disagrees

    n_raters = 3
    N = len(grouped)
    if N == 0:
        return float("nan")

    Pi_list = []
    for i, row in grouped.iterrows():
        agrees = row["sum"]
        disagrees = row["count"] - agrees
        Pi = (agrees**2 + disagrees**2 - n_raters) / (n_raters*(n_raters-1))
        Pi_list.append(Pi)

    P_bar = np.mean(Pi_list)
    total_agrees = df_field["AgreeBinary"].sum()
    total_responses = len(df_field)
    p_agree = total_agrees / total_responses
    p_disagree = 1.0 - p_agree
    Pe = p_agree**2 + p_disagree**2

    if abs(1 - Pe) < 1e-9:
        return float("nan")
    kappa = (P_bar - Pe) / (1 - Pe)
    return kappa

def majority_vote_agree(df_field_case: pd.DataFrame) -> str:
    """
    For a single field+case with 3 rows (one per rater),
    return "Agree" if >=2 say "Agree", else "Disagree".
    """
    agrees = (df_field_case["ReviewerResponse"].str.lower() == "agree").sum()
    return "Agree" if agrees >= 2 else "Disagree"

def get_tp_fp_tn_fn(llm_value: str, majority_label: str):
    """
    If LLM_value != "Not inferred" => predicted positive,
    else => predicted negative.
    If majority_label = "Agree" => ground truth positive,
    else => ground truth negative.
    
    Return (TP, FP, TN, FN).
    """
    guessed_positive = bool(llm_value) and llm_value.strip().lower() != "not inferred"
    ground_truth_positive = (majority_label.strip().lower() == "agree")

    if guessed_positive and ground_truth_positive:
        return (1, 0, 0, 0)
    elif guessed_positive and not ground_truth_positive:
        return (0, 1, 0, 0)
    elif not guessed_positive and not ground_truth_positive:
        return (0, 0, 1, 0)
    else:
        return (0, 0, 0, 1)

###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Extractions from Google Form CSVs")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to store output CSV files (default: current directory)."
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # A) Download & parse all forms
    all_parsed = []
    for form_name, csv_url in DATA_SOURCES.items():
        print(f"Loading form: {form_name} from {csv_url}")
        df_raw = pd.read_csv(csv_url)
        parsed_df = parse_single_form(df_raw, form_name)
        all_parsed.append(parsed_df)

    master_df = pd.concat(all_parsed, ignore_index=True)
    master_csv_path = os.path.join(output_dir, "master_long_data.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"=> Saved combined long data to {master_csv_path}")

    # B) Compute Fleiss' Kappa by field
    fleiss_results = []
    for fld in EXPECTED_FIELDS:
        df_fld = master_df[master_df["FieldName"] == fld]
        if len(df_fld) == 0:
            continue
        kappa_val = compute_fleiss_kappa_for_field(df_fld)
        fleiss_results.append({"FieldName": fld, "FleissKappa": kappa_val})

    df_kappa = pd.DataFrame(fleiss_results)
    kappa_csv_path = os.path.join(output_dir, "fleiss_kappa_by_field.csv")
    df_kappa.to_csv(kappa_csv_path, index=False)
    print(f"=> Saved Fleiss' Kappa results to {kappa_csv_path}")

    # C) Confusion Matrix (TP, FP, TN, FN) using majority vote
    grouped = master_df.groupby(["FormName", "CaseIndex", "FieldName"], as_index=False)
    cm_records = []
    for (fm, cs, fld), subdf in grouped:
        llm_value = subdf["LLM_Value"].iloc[0]  # same for all 3 rows in practice
        maj_label = majority_vote_agree(subdf)
        tp, fp, tn, fn = get_tp_fp_tn_fn(llm_value, maj_label)
        cm_records.append({
            "FormName": fm,
            "CaseIndex": cs,
            "FieldName": fld,
            "LLM_Value": llm_value,
            "MajorityLabel": maj_label,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
    df_cm = pd.DataFrame(cm_records)
    cm_csv_path = os.path.join(output_dir, "confusion_matrix_raw.csv")
    df_cm.to_csv(cm_csv_path, index=False)
    print(f"=> Saved confusion matrix (raw) to {cm_csv_path}")

    # Summarize by field
    summary_cm = df_cm.groupby("FieldName")[["TP","FP","TN","FN"]].sum().reset_index()

    def metrics(row):
        TP = row["TP"]; FP = row["FP"]; TN = row["TN"]; FN = row["FN"]
        total = TP+FP+TN+FN
        accuracy = (TP + TN) / total if total > 0 else np.nan
        precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        if precision is not np.nan and recall is not np.nan and precision+recall>0:
            f1 = 2*precision*recall/(precision+recall)
        else:
            f1 = np.nan
        return pd.Series({"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1":f1})

    metrics_df = summary_cm.apply(metrics, axis=1)
    final_metrics = pd.concat([summary_cm, metrics_df], axis=1)
    metrics_csv_path = os.path.join(output_dir, "extraction_metrics_by_field.csv")
    final_metrics.to_csv(metrics_csv_path, index=False)
    print(f"=> Saved extraction metrics by field to {metrics_csv_path}")

    # D) Extract free-text comments
    comment_groups = master_df.groupby(["FormName","CaseIndex","ReviewerEmail"], as_index=False)
    comment_records = []
    for (fm, cs, rv), subdf in comment_groups:
        cvals = subdf["Comment"].dropna().unique()
        comment_str = ""
        if len(cvals) > 0:
            comment_str = " | ".join(map(str, cvals))
        comment_records.append({
            "FormName": fm,
            "CaseIndex": cs,
            "ReviewerEmail": rv,
            "Comment": comment_str
        })
    df_comments = pd.DataFrame(comment_records)
    comments_csv_path = os.path.join(output_dir, "reviewer_comments.csv")
    df_comments.to_csv(comments_csv_path, index=False)
    print(f"=> Saved reviewer comments to {comments_csv_path}")

    # E) Extract LLM satisfaction ratings
    rating_records = []
    for (fm, cs, rv), subdf in comment_groups:
        rvals = subdf["Rating"].dropna().unique()
        rating_val = rvals[0] if len(rvals) > 0 else None
        rating_records.append({
            "FormName": fm,
            "CaseIndex": cs,
            "ReviewerEmail": rv,
            "Rating": rating_val
        })
    df_ratings = pd.DataFrame(rating_records)
    ratings_csv_path = os.path.join(output_dir, "llm_ratings.csv")
    df_ratings.to_csv(ratings_csv_path, index=False)
    print(f"=> Saved LLM ratings to {ratings_csv_path}")

    # F) Time analysis
    time_groups = master_df.groupby(["FormName","ReviewerEmail"], as_index=False)
    time_records = []
    for (fm, rv), subdf in time_groups:
        tvals = subdf["TimeToComplete"].dropna().unique()
        if len(tvals) == 0:
            time_val = None
        else:
            time_val = tvals[0]
        num_cases = subdf["CaseIndex"].nunique()
        time_records.append({
            "FormName": fm,
            "ReviewerEmail": rv,
            "TimeMinutesForWholeForm": time_val,
            "NumCasesInForm": num_cases,
        })

    df_times = pd.DataFrame(time_records)
    def compute_avg_time(row):
        if pd.isna(row["TimeMinutesForWholeForm"]) or row["NumCasesInForm"]==0:
            return None
        return float(row["TimeMinutesForWholeForm"])/row["NumCasesInForm"]
    df_times["AvgTimePerCase"] = df_times.apply(compute_avg_time, axis=1)

    times_csv_path = os.path.join(output_dir, "reviewer_times_raw.csv")
    df_times.to_csv(times_csv_path, index=False)
    print(f"=> Saved reviewer time data to {times_csv_path}")

    overall_times = df_times.dropna(subset=["AvgTimePerCase"])["AvgTimePerCase"].values
    if len(overall_times) > 0:
        mean_time = np.mean(overall_times)
        sd_time = np.std(overall_times, ddof=1)
        n = len(overall_times)
        sem = sd_time / math.sqrt(n)
        z_95 = 1.96
        lower = mean_time - z_95*sem
        upper = mean_time + z_95*sem

        print(f"Overall average time per case = {mean_time:.2f} min (95% CI: {lower:.2f}–{upper:.2f}), n={n}")
        summary_time = pd.DataFrame([{
            "MeanTimePerCase": mean_time,
            "SD": sd_time,
            "N": n,
            "CI_95_Lower": lower,
            "CI_95_Upper": upper,
        }])
        time_summary_path = os.path.join(output_dir, "time_per_case_summary.csv")
        summary_time.to_csv(time_summary_path, index=False)
        print(f"=> Saved summary time stats to {time_summary_path}")
    else:
        print("[INFO] No time data found for computing averages.")

    print("Analysis complete!")

if __name__ == "__main__":
    main()


# Execute script

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/evaluate_llm_extraction.py --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Clinician_Evaluation/tmp
