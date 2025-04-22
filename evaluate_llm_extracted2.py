#!/usr/bin/env python3
# evaluate_llm_extractions.py

"""
Evaluates LLM Extractions from clinician forms:
1) Finds 'Time to complete' column by name (not just last col).
2) Uses user-specified TP/FP/TN/FN logic => confusion_matrix_raw & extraction_metrics_by_field.
3) Fleiss' Kappa (2-cat) => <field>_debug_per_rater.csv. Unify "marie.duclosro" => "marie.duclos.ro".
4) Then we produce confusion_matrix_overall, LLM ratings, time analysis.
5) After we create <field>_debug_per_rater.csv, 
   we make <field>_debug_per_rater_modified.csv by mapping (FormName, oldCaseIndex) => a single global NewCaseIndex
   that continues across forms (no restart at 1).
6) We pivot that => <field>_cases_wide.csv, with actual email addresses as column names.

Usage:
  python evaluate_llm_extractions.py --output_dir /path/to/results

Requires:
  pip install statsmodels
"""

import argparse
import os
import re
import math

import pandas as pd
import numpy as np
import statsmodels.stats.inter_rater as ir

from data_sources import DATA_SOURCES

###############################################################################
# 1) CONFIG: EXPECTED FIELDS
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
# 2) EMAIL NORMALIZATION
###############################################################################
def normalize_email(email: str) -> str:
    """
    Merge 'marie.duclosro@gmail.com' => 'marie.duclos.ro@gmail.com'.
    """
    e = email.strip().lower()
    if e in ["marie.duclosro@gmail.com","marie.duclos.ro@gmail.com"]:
        return "marie.duclos.ro@gmail.com"
    return e

###############################################################################
# 3) COL PARSING
###############################################################################
def extract_field_from_colname(colname: str) -> str:
    match = re.search(r"\[(.*?)\]", colname)
    if not match:
        return None
    bracket_content = match.group(1)
    parts = bracket_content.split(":")
    if len(parts)<2:
        return None
    candidate_field = parts[0].strip()
    if candidate_field in EXPECTED_FIELDS:
        return candidate_field
    return None

def extract_llm_value(colname: str) -> str:
    match = re.search(r"\[(.*?)\]", colname)
    if not match:
        return None
    bracket_content = match.group(1)
    parts = bracket_content.split(":")
    if len(parts)<2:
        return None
    return parts[1].strip()

def find_time_column(df: pd.DataFrame) -> str:
    possible=[]
    for c in df.columns:
        lc = c.lower()
        if "time" in lc and "complete" in lc:
            possible.append(c)
    if len(possible)==1:
        return possible[0]
    elif len(possible)>1:
        print(f"[INFO] multiple time columns => {possible}, using {possible[0]}")
        return possible[0]
    return None

###############################################################################
# 4) PARSE SINGLE FORM => LONG
###############################################################################
def parse_single_form(df: pd.DataFrame, form_name: str) -> pd.DataFrame:
    """
    Convert wide => long, each row => 1 field from 1 case from 1 reviewer.
    We unify 'marie.duclosro' => 'marie.duclos.ro' right away.
    """
    time_col = find_time_column(df)
    if time_col:
        try:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        except:
            pass
    else:
        print(f"[WARNING] {form_name}: no time => TimeToComplete=None")

    skip_cols = set(df.columns[:2])  # col0=Timestamp, col1=Email
    if time_col:
        skip_cols.add(time_col)

    data_cols = [c for c in df.columns if c not in skip_cols]
    long_records=[]

    for i in range(len(df)):
        row_data= df.iloc[i]
        raw_email= str(row_data.iloc[1])
        reviewer_email= normalize_email(raw_email)
        time_val= row_data[time_col] if time_col else None

        chunk_size=32
        n_data= len(data_cols)
        n_cases= n_data//chunk_size
        leftover= n_data%chunk_size
        if leftover>0:
            print(f"[INFO] {form_name}: leftover={leftover}, ignoring partial chunk.")
        idx=0
        for case_i in range(1, n_cases+1):
            chunk= data_cols[idx: idx+chunk_size]
            idx+=chunk_size
            fields_part= chunk[:30]
            comment_col= chunk[30] if len(chunk)>30 else None
            rating_col= chunk[31] if len(chunk)>31 else None

            comment_val= row_data[comment_col] if comment_col else None
            rating_val= row_data[rating_col]   if rating_col  else None

            for cfield in fields_part:
                val= row_data[cfield]
                field_name= extract_field_from_colname(cfield)
                llm_val   = extract_llm_value(cfield)
                if not field_name:
                    continue
                rec = {
                    "FormName": form_name,
                    "ReviewerEmail": reviewer_email,
                    "CaseIndex": case_i,
                    "FieldName": field_name,
                    "LLM_Value": llm_val,
                    "ReviewerResponse": str(val).strip() if pd.notna(val) else "",
                    "Comment": comment_val,
                    "Rating": rating_val,
                    "TimeToComplete": time_val
                }
                long_records.append(rec)

    return pd.DataFrame(long_records)

###############################################################################
# 5) FLEISS' KAPPA => <field>_debug_per_rater.csv
###############################################################################
def fleiss_kappa_agree_disagree(df_field: pd.DataFrame, output_dir: str, field_name: str):
    """
    We unify marie.* emails, build Nx2 => fleiss_kappa. Also produce <field>_debug_per_rater.csv in InterRater folder.
    This debug CSV includes FormName, old CaseIndex, ReviewerEmail, etc. for the next step.
    """
    df_field = df_field.copy()
    df_field["ReviewerEmail"] = df_field["ReviewerEmail"].apply(normalize_email)
    df_field["IsAgree"] = df_field["ReviewerResponse"].apply(lambda x: 1 if x.lower()=="agree" else 0)

    debug_dir= os.path.join(output_dir,"InterRater")
    os.makedirs(debug_dir, exist_ok=True)
    debug_csv= os.path.join(debug_dir, f"{field_name}_debug_per_rater.csv")
    # we also keep FormName for the next step, to unify global indexing
    df_dbg= df_field[["FormName","CaseIndex","ReviewerEmail","FieldName","ReviewerResponse","IsAgree"]]
    df_dbg.to_csv(debug_csv, index=False)
    print(f"[DEBUG] Wrote {debug_csv}")

    # group => Nx2
    grouped= df_field.groupby("CaseIndex")["IsAgree"].agg(["sum","count"]).reset_index()
    data=[]
    for _, row in grouped.iterrows():
        agrees= int(row["sum"])
        total= int(row["count"])
        disagrees= total- agrees
        data.append([disagrees, agrees])
    arr= np.array(data)
    if len(arr)<1:
        return np.nan

    table,cats= ir.aggregate_raters(arr)
    kappa= ir.fleiss_kappa(table)
    return kappa

###############################################################################
# 6) CONFUSION MATRIX
###############################################################################
def interpret_llm_prediction(llm_value: str) -> bool:
    if not isinstance(llm_value, str):
        return False
    return llm_value.strip().lower()!="not inferred"

def get_tp_fp_tn_fn(pred_positive: bool, majority_says_correct: bool):
    if pred_positive and majority_says_correct:
        return (1,0,0,0)
    elif pred_positive and not majority_says_correct:
        return (0,1,0,0)
    elif not pred_positive and majority_says_correct:
        return (0,0,1,0)
    else:
        return (0,0,0,1)

###############################################################################
# 7) MAKE MODIFIED & WIDE => single global indexing across forms
###############################################################################
def make_modified_and_wide(debug_csv_path: str, field_name: str, output_dir: str):
    """
    1) read <field>_debug_per_rater.csv (which has FormName,CaseIndex,ReviewerEmail,IsAgree,...)
    2) unify (FormName,CaseIndex) => a single global NewCaseIndex. 
       e.g. if the file has (FormName=Group1,CaseIndex=1..5) then (Group2,CaseIndex=1..5), 
       we combine => 1..10, continuing across forms, in the order they appear in the CSV.

    3) save as <field>_debug_per_rater_modified.csv => columns: FormName, OldCaseIndex, NewCaseIndex, 
       plus ReviewerEmail, IsAgree, etc.
    4) pivot => <field>_cases_wide.csv => index=NewCaseIndex, columns=ReviewerEmail, values=IsAgree
    """
    if not os.path.exists(debug_csv_path):
        print(f"[POST] No debug CSV => {debug_csv_path}")
        return

    df = pd.read_csv(debug_csv_path)
    if df.empty:
        print(f"[POST] {debug_csv_path} is empty => skipping wide table creation.")
        return

    # unify marie
    df["ReviewerEmail"] = df["ReviewerEmail"].apply(normalize_email)

    # we'll define a "key" as (FormName, CaseIndex)
    # to preserve the order they appear, we'll do a unique, in the order of occurrence
    # then map them => NewCaseIndex=1..N
    # e.g. if the file lines are (Group1,CaseIndex=1) then (Group1,CaseIndex=2) ... (Group2,CaseIndex=1) => continuing
    # We'll do:
    df["_combined_key"] = df["FormName"].astype(str) + "||" + df["CaseIndex"].astype(str)

    # we want to find unique keys in the order of first appearance
    # a trick is to do df.drop_duplicates(subset=["_combined_key"])
    # then enumerate
    unique_keys = []
    seen = set()
    for idx,row in df.iterrows():
        ck = row["_combined_key"]
        if ck not in seen:
            seen.add(ck)
            unique_keys.append(ck)

    # now build map
    key_map = {}
    next_idx = 1
    for ck in unique_keys:
        key_map[ck] = next_idx
        next_idx += 1

    df["NewCaseIndex"] = df["_combined_key"].map(key_map)
    # This ensures if first form had CaseIndex=1..5 => we get 1..5
    # second form's CaseIndex=1..5 => => next 6..10, etc.

    # we can now drop the _combined_key
    df.drop(columns=["_combined_key"], inplace=True)

    # save as <field>_debug_per_rater_modified.csv
    mod_csv= os.path.join(output_dir, f"{field_name}_debug_per_rater_modified.csv")
    df.to_csv(mod_csv, index=False)
    print(f"[POST] Saved => {mod_csv} with single global NewCaseIndex across forms.")

    # pivot => wide
    wide = df.pivot_table(
        index="NewCaseIndex",
        columns="ReviewerEmail",
        values="IsAgree",
        aggfunc="first"
    )
    wide.reset_index(inplace=True)

    wide_csv= os.path.join(output_dir, f"{field_name}_cases_wide.csv")
    wide.to_csv(wide_csv, index=False)
    print(f"[POST] Created wide table => {wide_csv} with {len(wide)} rows")


###############################################################################
# MAIN
###############################################################################
def main():
    parser= argparse.ArgumentParser(description="Evaluate LLM Extractions + unify emails + single global indexing.")
    parser.add_argument("--output_dir", default=".", help="Where to save CSVs.")
    args= parser.parse_args()

    outdir= args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # A) parse forms => unify marie
    all_parsed=[]
    for form_name, csv_url in DATA_SOURCES.items():
        print(f"[INFO] {form_name} => {csv_url}")
        df_raw= pd.read_csv(csv_url)
        parsed_df= parse_single_form(df_raw, form_name)
        all_parsed.append(parsed_df)

    master_df = pd.concat(all_parsed, ignore_index=True)
    master_df["ReviewerEmail"] = master_df["ReviewerEmail"].apply(normalize_email)

    master_csv= os.path.join(outdir,"master_long_data.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"=> Saved master_long_data.csv => {master_csv}")

    # B) Fleiss' Kappa
    fleiss_list=[]
    for fld in EXPECTED_FIELDS:
        df_fld = master_df[master_df["FieldName"]==fld].copy()
        if len(df_fld)==0:
            continue
        kappa_val = fleiss_kappa_agree_disagree(df_fld, outdir, fld)
        fleiss_list.append({"FieldName":fld, "FleissKappa":kappa_val})
    df_kappa= pd.DataFrame(fleiss_list).sort_values("FieldName").reset_index(drop=True)
    kappa_csv= os.path.join(outdir,"fleiss_kappa_by_field.csv")
    df_kappa.to_csv(kappa_csv, index=False)
    print(f"=> Saved fleiss_kappa_by_field => {kappa_csv}")

    # C) confusion matrix
    grouped= master_df.groupby(["FormName","CaseIndex","FieldName"], as_index=False)
    cm_rows=[]
    for (fm,cs,fld), subdf in grouped:
        num_agree= (subdf["ReviewerResponse"].str.lower()=="agree").sum()
        majority_correct= (num_agree>=2)
        llm_val= subdf["LLM_Value"].iloc[0]
        pred_pos= interpret_llm_prediction(llm_val)
        tp,fp,tn,fn= get_tp_fp_tn_fn(pred_pos, majority_correct)
        cm_rows.append({
            "FormName":fm,"CaseIndex":cs,"FieldName":fld,
            "LLM_Value":llm_val,
            "MajoritySaysCorrect": majority_correct,
            "TP":tp,"FP":fp,"TN":tn,"FN":fn
        })
    df_cm= pd.DataFrame(cm_rows)
    cm_raw= os.path.join(outdir,"confusion_matrix_raw.csv")
    df_cm.to_csv(cm_raw, index=False)
    print(f"=> Saved confusion_matrix_raw => {cm_raw}")

    # Summarize by field
    sum_fields= df_cm.groupby("FieldName")[["TP","FP","TN","FN"]].sum().reset_index()
    def calc_metrics(row):
        TP= row["TP"]; FP= row["FP"]; TN= row["TN"]; FN= row["FN"]
        total= TP+FP+TN+FN
        acc= (TP+TN)/total if total>0 else np.nan
        prec= TP/(TP+FP) if (TP+FP)>0 else np.nan
        rec= TP/(TP+FN) if (TP+FN)>0 else np.nan
        f1= (2*prec*rec)/(prec+rec) if prec is not np.nan and rec is not np.nan and (prec+rec)>0 else np.nan
        return pd.Series({"TP":TP,"FP":FP,"TN":TN,"FN":FN,
                          "Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1})
    df_field_metrics= sum_fields.apply(calc_metrics, axis=1)
    df_field_metrics= pd.concat([sum_fields["FieldName"], df_field_metrics], axis=1)
    field_csv= os.path.join(outdir,"extraction_metrics_by_field.csv")
    df_field_metrics.to_csv(field_csv, index=False)
    print(f"=> Saved extraction_metrics_by_field => {field_csv}")

    # Overall
    overall= df_cm[["TP","FP","TN","FN"]].sum()
    TP= overall["TP"]; FP= overall["FP"]
    TN= overall["TN"]; FN= overall["FN"]
    total= TP+FP+TN+FN
    acc= (TP+TN)/total if total>0 else np.nan
    prec= TP/(TP+FP) if (TP+FP)>0 else np.nan
    rec= TP/(TP+FN) if (TP+FN)>0 else np.nan
    f1= (2*prec*rec)/(prec+rec) if prec is not np.nan and rec is not np.nan and (prec+rec)>0 else np.nan
    df_overall= pd.DataFrame([{
        "TP":TP,"FP":FP,"TN":TN,"FN":FN,
        "Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1
    }])
    overall_csv= os.path.join(outdir,"confusion_matrix_overall.csv")
    df_overall.to_csv(overall_csv, index=False)
    print(f"=> Saved confusion_matrix_overall => {overall_csv}")

    # D) LLM Ratings
    df_ratings= master_df[["FormName","CaseIndex","ReviewerEmail","Rating"]].drop_duplicates()
    rating_raw= os.path.join(outdir,"llm_ratings_raw.csv")
    df_ratings.to_csv(rating_raw, index=False)
    print(f"=> Saved llm_ratings_raw => {rating_raw}")

    df_ratings["NumericRating"]= pd.to_numeric(df_ratings["Rating"], errors="coerce")
    rev_stats= df_ratings.groupby("ReviewerEmail")["NumericRating"].agg(["count","mean","std"]).reset_index()
    rev_stats.rename(columns={"count":"N_cases","mean":"MeanRating","std":"SDRating"}, inplace=True)
    rating_summ= os.path.join(outdir,"llm_ratings_summary.csv")
    rev_stats.to_csv(rating_summ, index=False)
    print(f"=> Saved llm_ratings_summary => {rating_summ}")

    # E) Times
    df_time= master_df[["FormName","ReviewerEmail","CaseIndex","TimeToComplete"]].drop_duplicates()
    time_groups= df_time.groupby(["FormName","ReviewerEmail"], as_index=False)
    recs=[]
    for (fm,rv), subdf2 in time_groups:
        tvals= subdf2["TimeToComplete"].dropna().unique()
        if len(tvals)==1:
            tval= tvals[0]
        elif len(tvals)>1:
            tval= tvals[0]
            print(f"[WARNING] multiple times => {fm}-{rv}: {tvals} => picking {tval}")
        else:
            tval= None
        n_cases= subdf2["CaseIndex"].nunique()
        recs.append({
            "FormName":fm,"ReviewerEmail":rv,
            "TimeMinutesForWholeForm":tval,
            "NumCasesInForm":n_cases
        })
    df_times= pd.DataFrame(recs)
    def avgtime(row):
        if pd.isna(row["TimeMinutesForWholeForm"]) or row["NumCasesInForm"]==0:
            return None
        return float(row["TimeMinutesForWholeForm"])/row["NumCasesInForm"]
    df_times["AvgTimePerCase"]= df_times.apply(avgtime, axis=1)
    times_raw= os.path.join(outdir,"reviewer_times_raw.csv")
    df_times.to_csv(times_raw, index=False)
    print(f"=> Saved reviewer_times_raw => {times_raw}")

    arr_times= df_times.dropna(subset=["AvgTimePerCase"])["AvgTimePerCase"].values
    if len(arr_times)>0:
        mean_t= np.mean(arr_times)
        sd_t= np.std(arr_times, ddof=1)
        n= len(arr_times)
        sem= sd_t/math.sqrt(n) if n>1 else 0
        z95=1.96
        lower= mean_t- z95*sem
        upper= mean_t+ z95*sem
        df_tsum= pd.DataFrame([{
            "MeanTimePerCase":mean_t,
            "SD":sd_t,
            "N":n,
            "CI_95_Lower":lower,
            "CI_95_Upper":upper
        }])
        time_sum= os.path.join(outdir,"time_per_case_summary.csv")
        df_tsum.to_csv(time_sum, index=False)
        print(f"=> Overall time/case= {mean_t:.2f} min (95% CI {lower:.2f}-{upper:.2f}), n={n}")
        print(f"=> Saved time_per_case_summary => {time_sum}")
    else:
        print("[INFO] No numeric time => skipping time summary.")

    print("Analysis complete!")

    # F) For each <field>_debug_per_rater.csv => build single global index from (FormName, oldCaseIndex)
    debug_dir= os.path.join(outdir,"InterRater")
    if not os.path.isdir(debug_dir):
        print("[INFO] No InterRater/ => skipping wide table creation.")
        return

    for fld in EXPECTED_FIELDS:
        debug_csv= os.path.join(debug_dir, f"{fld}_debug_per_rater.csv")
        if os.path.exists(debug_csv):
            make_modified_and_wide(debug_csv, fld, debug_dir)

def make_modified_and_wide(debug_csv_path: str, field_name: str, output_dir: str):
    """
    We produce:
      <field>_debug_per_rater_modified.csv => same data + 'NewCaseIndex' = single global across forms
        (based on the order (FormName,CaseIndex) appear)
      <field>_cases_wide.csv => pivot on 'NewCaseIndex' for the row, 'ReviewerEmail' for columns, storing 'IsAgree'.
    """
    if not os.path.exists(debug_csv_path):
        print(f"[POST] missing debug CSV => {debug_csv_path}")
        return
    df = pd.read_csv(debug_csv_path)
    if df.empty:
        print(f"[POST] {debug_csv_path} is empty => skipping wide table creation.")
        return

    # unify marie again
    df["ReviewerEmail"] = df["ReviewerEmail"].apply(normalize_email)

    # build combined key => (FormName,CaseIndex), to preserve order of first appearance
    df["_combined_key"] = df["FormName"].astype(str) + "||" + df["CaseIndex"].astype(str)
    unique_keys=[]
    seen=set()
    for i,row in df.iterrows():
        ck=row["_combined_key"]
        if ck not in seen:
            seen.add(ck)
            unique_keys.append(ck)
    # now map them => 1..N
    key_map={}
    next_idx=1
    for ck in unique_keys:
        key_map[ck] = next_idx
        next_idx+=1

    df["NewCaseIndex"] = df["_combined_key"].map(key_map)
    df.drop(columns=["_combined_key"], inplace=True)

    # Save modified
    mod_csv = os.path.join(output_dir,f"{field_name}_debug_per_rater_modified.csv")
    df.to_csv(mod_csv, index=False)
    print(f"[POST] Saved => {mod_csv} with single global index across forms as NewCaseIndex.")

    # pivot => <field>_cases_wide.csv
    wide = df.pivot_table(
        index="NewCaseIndex",
        columns="ReviewerEmail",
        values="IsAgree",
        aggfunc="first"
    )
    wide.reset_index(inplace=True)
    wide_csv= os.path.join(output_dir,f"{field_name}_cases_wide.csv")
    wide.to_csv(wide_csv, index=False)
    print(f"[POST] Created wide table => {wide_csv} with {len(wide)} rows")


if __name__=="__main__":
    main()


# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/evaluate_llm_extracted2.py --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Clinician_Evaluation/
