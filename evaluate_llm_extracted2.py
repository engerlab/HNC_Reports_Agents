#!/usr/bin/env python3
# evaluate_llm_extractions.py

"""
Evaluate LLM Extractions with:
1) A single global index (GlobalCaseIndex) across all forms, built from (FormName, oldCaseIndex).
2) Normalizes "marie.duclosro@gmail.com" => "marie.duclos.ro@gmail.com".
3) Groups by GlobalCaseIndex for Fleiss’ Kappa.
4) Creates <field>_cases_wide.csv, reordering columns as:
     [NewCaseIndex, georgeshenouda413@gmail.com, tomasyokoo@hotmail.com, marie.duclos.ro@gmail.com, ...]
   for usage in irrCAC.
5) Optionally runs irrCAC.raw.CAC(...) on each wide CSV, storing a text result.

Confusion matrix, rating analysis, time analysis also use the GlobalCaseIndex.

Usage:
  python evaluate_llm_extractions.py --output_dir /some/dir

Requires:
  pip install statsmodels irrCAC
"""

import argparse
import os
import re
import math

import pandas as pd
import numpy as np
import statsmodels.stats.inter_rater as ir
import krippendorff
import warnings

# If you plan to use irrCAC:
try:
    from irrCAC.raw import CAC
    HAVE_IRRCAC = True
except ImportError:
    HAVE_IRRCAC = False
    print("[INFO] irrCAC not installed. We'll skip the optional step for final Fleiss from wide CSV.")

from data_sources import DATA_SOURCES

###############################################################################
# 1) EXPECTED FIELDS
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
    candidate = parts[0].strip()
    if candidate in EXPECTED_FIELDS:
        return candidate
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
        low = c.lower()
        if "time" in low and "complete" in low:
            possible.append(c)
    if len(possible)==1:
        return possible[0]
    elif len(possible)>1:
        print(f"[INFO] multiple time columns => {possible}, picking {possible[0]}")
        return possible[0]
    return None

###############################################################################
# 4) PARSE SINGLE FORM => LONG
###############################################################################
def parse_single_form(df: pd.DataFrame, form_name: str) -> pd.DataFrame:
    """
    wide => long. We'll store oldCaseIndex=1..N for each form. We'll unify 'marie.*' immediately.
    """
    time_col = find_time_column(df)
    if time_col:
        try:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        except:
            pass
    else:
        print(f"[WARNING] {form_name}: no time => TimeToComplete=None.")

    skip_cols = set(df.columns[:2])  # col0=Timestamp, col1=Email
    if time_col:
        skip_cols.add(time_col)

    data_cols = [c for c in df.columns if c not in skip_cols]
    long_records=[]

    for i in range(len(df)):
        row = df.iloc[i]
        raw_email= str(row.iloc[1])
        reviewer_email= normalize_email(raw_email)
        time_val= row[time_col] if time_col else None

        chunk_size=32
        n_data= len(data_cols)
        n_cases= n_data//chunk_size
        leftover= n_data%chunk_size
        if leftover>0:
            print(f"[INFO] {form_name}: leftover {leftover}, ignoring partial chunk.")
        idx=0
        for cidx in range(1, n_cases+1):
            chunk= data_cols[idx : idx+chunk_size]
            idx+= chunk_size
            fields_part= chunk[:30]
            comment_col= chunk[30] if len(chunk)>30 else None
            rating_col= chunk[31] if len(chunk)>31 else None

            comment_val= row[comment_col] if comment_col else None
            rating_val= row[rating_col]   if rating_col  else None

            for field_col in fields_part:
                val= row[field_col]
                field_name= extract_field_from_colname(field_col)
                llm_val= extract_llm_value(field_col)
                if not field_name:
                    continue
                rec={
                    "FormName": form_name,
                    "oldCaseIndex": cidx,
                    "ReviewerEmail": reviewer_email,
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
# 5) BUILD SINGLE GLOBALCASEINDEX
###############################################################################
def build_global_index(master_df: pd.DataFrame):
    """
    Combine (FormName, oldCaseIndex) => single GlobalCaseIndex across all forms 
    in the order they appear in 'master_df'.
    """
    master_df["_combined_key"] = master_df["FormName"].astype(str) + "||" + master_df["oldCaseIndex"].astype(str)
    unique_keys=[]
    seen=set()
    for i,row in master_df.iterrows():
        ck=row["_combined_key"]
        if ck not in seen:
            seen.add(ck)
            unique_keys.append(ck)

    key_map={}
    next_idx=1
    for ck in unique_keys:
        key_map[ck]= next_idx
        next_idx+=1

    master_df["GlobalCaseIndex"] = master_df["_combined_key"].map(key_map)
    master_df.drop(columns=["_combined_key"], inplace=True)

###############################################################################
# 6) FLEISS' KAPPA => group by GlobalCaseIndex
###############################################################################
def fleiss_kappa_by_field(master_df: pd.DataFrame, output_dir: str):
    """
    For each field => group by GlobalCaseIndex => Nx2 => fleiss_kappa.
    Then produce <field>_debug_per_rater.csv => columns: 
        [FormName, oldCaseIndex, GlobalCaseIndex, ReviewerEmail, FieldName, ReviewerResponse, IsAgree].
    """
    debug_dir= os.path.join(output_dir,"InterRater")
    os.makedirs(debug_dir, exist_ok=True)
    results=[]
    for fld in EXPECTED_FIELDS:
        sub= master_df[master_df["FieldName"]==fld].copy()
        if sub.empty:
            continue
        sub["IsAgree"]= sub["ReviewerResponse"].str.lower().eq("agree").astype(int)
        # debug CSV
        debug_csv= os.path.join(debug_dir, f"{fld}_debug_per_rater.csv")
        dbg_cols = [
            "FormName","oldCaseIndex","GlobalCaseIndex","ReviewerEmail",
            "FieldName","ReviewerResponse","IsAgree"
        ]
        sub[dbg_cols].to_csv(debug_csv, index=False)
        print(f"[DEBUG] Wrote {debug_csv}")

        # Nx2 => fleiss
        grouped= sub.groupby("GlobalCaseIndex")["IsAgree"].agg(["sum","count"]).reset_index()
        if len(grouped)==0:
            results.append({"FieldName":fld,"FleissKappa":np.nan})
            continue
        arr=[]
        for _, row in grouped.iterrows():
            agrees= row["sum"]
            total= row["count"]
            disagrees= total- agrees
            arr.append([disagrees, agrees])
        arr= np.array(arr)
        if arr.size<1:
            results.append({"FieldName":fld,"FleissKappa":np.nan})
            continue
        table, cats = ir.aggregate_raters(arr)
        kappa_val= ir.fleiss_kappa(table)
        results.append({"FieldName":fld,"FleissKappa":kappa_val})

    df_kappa= pd.DataFrame(results)
    df_kappa.sort_values("FieldName", ignore_index=True, inplace=True)
    kappa_csv= os.path.join(output_dir,"fleiss_kappa_by_field.csv")
    df_kappa.to_csv(kappa_csv, index=False)
    print(f"=> Saved fleiss_kappa_by_field => {kappa_csv}")

###############################################################################
# 7) CONFUSION MATRIX => group by (GlobalCaseIndex, FieldName)
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

def compute_confusion_matrix(master_df: pd.DataFrame, output_dir: str):
    grouped= master_df.groupby(["GlobalCaseIndex","FieldName"], as_index=False)
    recs=[]
    for (gc,fld), sub in grouped:
        n_agree= sub["ReviewerResponse"].str.lower().eq("agree").sum()
        majority_correct= (n_agree>=2)
        llm_val= sub["LLM_Value"].iloc[0]
        pred_pos= interpret_llm_prediction(llm_val)
        tp,fp,tn,fn= get_tp_fp_tn_fn(pred_pos, majority_correct)
        recs.append({
            "GlobalCaseIndex":gc,
            "FieldName":fld,
            "LLM_Value":llm_val,
            "MajoritySaysCorrect":majority_correct,
            "TP":tp,"FP":fp,"TN":tn,"FN":fn
        })

    df_cm= pd.DataFrame(recs)
    cm_raw= os.path.join(output_dir,"confusion_matrix_raw.csv")
    df_cm.to_csv(cm_raw, index=False)
    print(f"=> Saved confusion_matrix_raw => {cm_raw}")

    # Summarize => per-field
    summ= df_cm.groupby("FieldName")[["TP","FP","TN","FN"]].sum().reset_index()
    def calc_metrics(row):
        TP= row["TP"]; FP= row["FP"]; TN= row["TN"]; FN= row["FN"]
        total= TP+FP+TN+FN
        acc= (TP+TN)/total if total>0 else np.nan
        prec= TP/(TP+FP) if (TP+FP)>0 else np.nan
        rec= TP/(TP+FN) if (TP+FN)>0 else np.nan
        f1= (2*prec*rec)/(prec+rec) if prec is not np.nan and rec is not np.nan and (prec+rec)>0 else np.nan
        return pd.Series({"TP":TP,"FP":FP,"TN":TN,"FN":FN,"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1})
    df_field= summ.apply(calc_metrics, axis=1)
    df_field= pd.concat([summ["FieldName"],df_field], axis=1)
    field_csv= os.path.join(output_dir,"extraction_metrics_by_field.csv")
    df_field.to_csv(field_csv, index=False)
    print(f"=> Saved extraction_metrics_by_field => {field_csv}")

    # Overall
    tot= df_cm[["TP","FP","TN","FN"]].sum()
    TP= tot["TP"]; FP= tot["FP"]
    TN= tot["TN"]; FN= tot["FN"]
    total= TP+FP+TN+FN
    acc= (TP+TN)/total if total>0 else np.nan
    prec= TP/(TP+FP) if (TP+FP)>0 else np.nan
    rec= TP/(TP+FN) if (TP+FN)>0 else np.nan
    f1= (2*prec*rec)/(prec+rec) if prec is not np.nan and rec is not np.nan and (prec+rec)>0 else np.nan
    df_overall= pd.DataFrame([{
        "TP":TP,"FP":FP,"TN":TN,"FN":FN,
        "Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1
    }])
    overall_csv= os.path.join(output_dir,"confusion_matrix_overall.csv")
    df_overall.to_csv(overall_csv, index=False)
    print(f"=> Saved confusion_matrix_overall => {overall_csv}")

###############################################################################
# 8) RATINGS & TIMES => referencing GlobalCaseIndex
###############################################################################
def analyze_ratings_times(master_df: pd.DataFrame, output_dir: str):
    # LLM ratings
    df_ratings= master_df[["GlobalCaseIndex","ReviewerEmail","Rating"]].drop_duplicates()
    rating_raw= os.path.join(output_dir,"llm_ratings_raw.csv")
    df_ratings.to_csv(rating_raw, index=False)
    print(f"=> Saved llm_ratings_raw => {rating_raw}")

    df_ratings["NumericRating"]= pd.to_numeric(df_ratings["Rating"], errors="coerce")
    rev_stats= df_ratings.groupby("ReviewerEmail")["NumericRating"].agg(["count","mean","std"]).reset_index()
    rev_stats.rename(columns={"count":"N_cases","mean":"MeanRating","std":"SDRating"}, inplace=True)
    rating_sum= os.path.join(output_dir,"llm_ratings_summary.csv")
    rev_stats.to_csv(rating_sum, index=False)
    print(f"=> Saved llm_ratings_summary => {rating_sum}")

    # Times
    df_time= master_df[["GlobalCaseIndex","ReviewerEmail","TimeToComplete"]].drop_duplicates()
    grouped= df_time.groupby("ReviewerEmail", as_index=False)
    recs=[]
    for rv, subd in grouped:
        tvals= subd["TimeToComplete"].dropna().unique()
        if len(tvals)==1:
            tval= tvals[0]
        elif len(tvals)>1:
            tval= tvals[0]
            print(f"[WARNING] multiple times => {rv}: {tvals} => picking first={tval}")
        else:
            tval= None
        n_cases= subd["GlobalCaseIndex"].nunique()
        recs.append({
            "ReviewerEmail":rv,
            "TimeMinutesForWholeForm":tval,
            "NumCasesInForm":n_cases
        })
    df_times= pd.DataFrame(recs)
    def avg_time(row):
        if pd.isna(row["TimeMinutesForWholeForm"]) or row["NumCasesInForm"]==0:
            return None
        return float(row["TimeMinutesForWholeForm"])/row["NumCasesInForm"]
    df_times["AvgTimePerCase"]= df_times.apply(avg_time, axis=1)
    times_raw= os.path.join(output_dir,"reviewer_times_raw.csv")
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
        time_sum= os.path.join(output_dir,"time_per_case_summary.csv")
        df_tsum.to_csv(time_sum, index=False)
        print(f"=> Overall time/case= {mean_t:.2f} (95%CI {lower:.2f}-{upper:.2f}), n={n}")
        print(f"=> Saved time_per_case_summary => {time_sum}")
    else:
        print("[INFO] No numeric time => skipping time summary.")

###############################################################################
# 9) OPTIONAL: compute irrCAC fleiss from <field>_cases_wide.csv
###############################################################################
def compute_irrCAC_from_wide(wide_csv:str, field_name:str):
    """
    If irrCAC is installed, read wide CSV => reorder columns => pass to CAC(...) => fleiss().
    We'll save the result to <field>_cases_wide_irrCAC.txt
    The user specifically wants columns in order:
      [GlobalCaseIndex, georgeshenouda413@gmail.com, tomasyokoo@hotmail.com, marie.duclos.ro@gmail.com]
    If a column doesn't exist, we skip it.
    """
    if not HAVE_IRRCAC:
        print("[INFO] irrCAC not installed => skipping compute_irrCAC_from_wide")
        return
    if not os.path.exists(wide_csv):
        print(f"[INFO] wide CSV not found => {wide_csv}")
        return
    df = pd.read_csv(wide_csv)
    if df.empty:
        print(f"[INFO] wide CSV {wide_csv} is empty => skipping.")
        return

    # reorder columns
    desired_order = [
        "GlobalCaseIndex",
        "georgeshenouda413@gmail.com",
        "tomasyokoo@hotmail.com",
        "marie.duclos.ro@gmail.com"
    ]
    # keep only existing ones in desired order:
    existing = [c for c in desired_order if c in df.columns]
    # Then add any extra columns not in the desired list
    extras = [c for c in df.columns if c not in existing and c!="GlobalCaseIndex"]
    # final col list:
    final_cols = existing + extras
    df = df[final_cols]

    # save back if you want to see the reordering
    df.to_csv(wide_csv, index=False)
    print(f"[POST] Reordered columns in => {wide_csv}")

    # Now read the actual data ignoring the first col => "GlobalCaseIndex"
    # so the rating matrix is df.iloc[:,1:]
    if df.shape[1] < 2:
        print("[WARNING] not enough columns for multiple raters => skipping irrCAC.")
        return

    data = df.iloc[:,1:].to_numpy()  # shape: (Ncases, Nraters)
    # If you have missing (NaN), you can decide how to handle them. 
    # irrCAC might fail. We'll try:
    from irrCAC.raw import CAC
    cac_obj = CAC(data)
    results = cac_obj.fleiss()
    # we'll store the string representation
    out_txt = wide_csv.replace(".csv","_irrCAC.txt")
    with open(out_txt,"w") as f:
        f.write("### irrCAC Fleiss results ###\n")
        f.write(str(results))
    print(f"[POST] Wrote irrCAC fleiss => {out_txt}")

def compute_irrCAC_from_wide(wide_csv:str, field_name:str):
    """
    Attempt irrCAC Fleiss on the final wide CSV.
    We'll skip the first col => "GlobalCaseIndex" => the rest is numeric data 
    (0 or 1 or possibly NaN).
    Save results to <field>_cases_wide_irrCAC.txt
    """
    df = pd.read_csv(wide_csv)
    if df.empty or df.shape[1]<2:
        print(f"[WARNING] {wide_csv} empty or only 1 column => skipping irrCAC fleiss.")
        return
    data = df.iloc[:,1:].to_numpy()  # skip GlobalCaseIndex
    # data needs to be a pandas DataFrame
    data = pd.DataFrame(data)
    print(f"[INFO] data shape for irrCAC: {data.shape}")
    print(f"[INFO] data: {data}")
    # save data to CSV for debugging
    debug_csv= wide_csv.replace(".csv","_irrCAC_debug.csv")
    df.to_csv(debug_csv, index=False)
    print(f"[POST] Wrote debug CSV for irrCAC => {debug_csv}")
    cac_obj = CAC(data)
    print(f"[INFO] cac_obj: {cac_obj}")
    fleiss = cac_obj.fleiss()
    print(f"[INFO] Fleiss results: {fleiss}")

    out_txt= wide_csv.replace(".csv","_irrCAC.txt")
    with open(out_txt,"w") as f:
        f.write("### irrCAC Fleiss results ###\n")
        f.write(str(fleiss))
    print(f"[POST] Wrote irrCAC fleiss => {out_txt}")
    # To calculate the Brennar-Prediger coefficient, call the bp() method.
    bp_results = cac_obj.bp()
    print(f"[INFO] Brennar-Prediger results: {bp_results}")
    # To calculate the Gwet's AC1, call the gwet() method.
    gwet = cac_obj.gwet()
    print(f"[INFO] Gwet's AC1 results: {gwet}")
    print(f"gwet['est']{gwet['est']['coefficient_value']}")
    # To calculate the Krippendorff's alpha, call the krippendorff() method.
    krippendorff = cac_obj.krippendorff()
    print(f"[INFO] Krippendorff's alpha results: {krippendorff}")
    # To calculate Conger's kappa, call the conger() method.
    conger = cac_obj.conger()
    print(f"[INFO] Conger's kappa results: {conger}")

###############################################################################
# 10) MAIN
###############################################################################
def main():

    parser= argparse.ArgumentParser(description="Single GlobalCaseIndex across forms + reorder columns in wide CSV.")
    parser.add_argument("--output_dir", default=".", help="Where to save CSVs")
    parser.add_argument("--run_irrCAC", action="store_true", help="If set, we attempt irrCAC fleiss on the final wide CSV.")
    args= parser.parse_args()

    outdir= args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # A) parse all forms
    all_parsed=[]
    for form_name, csv_url in DATA_SOURCES.items():
        print(f"[INFO] {form_name} => {csv_url}")
        df_raw= pd.read_csv(csv_url)
        parsed_df= parse_single_form(df_raw, form_name)
        all_parsed.append(parsed_df)
    master_df= pd.concat(all_parsed, ignore_index=True)
    # unify (already done in parse, but let's do again)
    master_df["ReviewerEmail"] = master_df["ReviewerEmail"].apply(normalize_email)

    # B) Build global index
    build_global_index(master_df)

    # save final master
    master_csv= os.path.join(outdir,"master_long_data.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"=> Saved master_long_data.csv => {master_csv}")

    # C) Fleiss => <field>_debug_per_rater.csv
    fleiss_kappa_by_field(master_df, outdir)

    # D) confusion => group by (GlobalCaseIndex, FieldName)
    compute_confusion_matrix(master_df, outdir)

    # E) rating/time
    analyze_ratings_times(master_df, outdir)

    # F) pivot each <field>_debug_per_rater.csv => <field>_cases_wide.csv => reorder columns
    debug_dir= os.path.join(outdir,"InterRater")
    if not os.path.isdir(debug_dir):
        print("[INFO] no InterRater folder => skipping wide pivot + irrCAC.")
        return

    for fld in EXPECTED_FIELDS:
        debug_csv= os.path.join(debug_dir, f"{fld}_debug_per_rater.csv")
        if not os.path.exists(debug_csv):
            continue
        df_dbg= pd.read_csv(debug_csv)
        if df_dbg.empty:
            continue
        # pivot => index=GlobalCaseIndex, columns=ReviewerEmail, values=IsAgree
        wide = df_dbg.pivot_table(
            index="GlobalCaseIndex",
            columns="ReviewerEmail",
            values="IsAgree",
            aggfunc="first"
        )
        wide.reset_index(inplace=True)
        # reorder columns => [GlobalCaseIndex, georgeshenouda, tomasyokoo, marie...], then any extras
        desired_cols = [
            "GlobalCaseIndex",
            "georgeshenouda413@gmail.com",
            "tomasyokoo@hotmail.com",
            "marie.duclos.ro@gmail.com"
        ]
        existing = [c for c in desired_cols if c in wide.columns]
        extras = [c for c in wide.columns if c not in existing and c!="GlobalCaseIndex"]
        final_cols = existing + extras
        wide = wide[final_cols]

        wide_csv= os.path.join(debug_dir, f"{fld}_cases_wide.csv")
        wide.to_csv(wide_csv, index=False)
        print(f"[POST] Created wide => {wide_csv} in requested col order: {final_cols}")

        # optionally run irrCAC on it
        if args.run_irrCAC and HAVE_IRRCAC:
            compute_irrCAC_from_wide(wide_csv, fld)
        elif args.run_irrCAC and not HAVE_IRRCAC:
            print("[WARNING] --run_irrCAC was set but irrCAC is not installed.")




if __name__=="__main__":
    main()


# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/evaluate_llm_extracted2.py --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Clinician_Evaluation/ --run_irrCAC
