# compute_metrics.py
import numpy as np
import pandas as pd
import math
from statsmodels.stats.inter_rater import fleiss_kappa

def compute_fleiss_kappa_3rater(df_parsed_all, CSV_URLS):
    """
    Computes the overall Fleiss' Kappa for groups that have 3 reviewers.
    (We treat each item=(group,case,field) with categories = {Agree,Disagree}).
    Returns the overall kappa value (float).
    """

    # Identify groups with 3 raters
    groups_3_raters = [g for g,v in CSV_URLS.items() if v["num_reviewers"] == 3]
    df_3 = df_parsed_all[df_parsed_all["group"].isin(groups_3_raters)].copy()

    # Binarize "Agree" => 1, else 0
    df_3["agree_bin"] = df_3["rater_agree_label"].apply(lambda x: 1 if str(x).lower().startswith("agree") else 0)

    # For each (group, case_index, field_index), count # of "agree", # of "disagree"
    item_groups = df_3.groupby(["group","case_index","field_index"])
    fleiss_data = []
    for (g,c,f), sub in item_groups:
        n_agree = sub["agree_bin"].sum()
        n_disagree = 3 - n_agree
        fleiss_data.append([n_agree, n_disagree])

    fleiss_array = np.array(fleiss_data)
    overall_kappa = fleiss_kappa(fleiss_array)
    return overall_kappa


def compute_confusion_matrix(df_parsed_all, CSV_URLS):
    """
    1) Determine LLM_label => 1 if llm_value_inferred=True, else 0.
    2) Determine ground_truth => majority for 3-rater groups, or first row if 2-rater group.
    3) Build confusion matrix (TP,FP,TN,FN), then compute accuracy, precision, recall, F1.
    Returns a dict with all these metrics.
    """
    df = df_parsed_all.copy()
    # LLM_label
    df["LLM_label"] = df["llm_value_inferred"].apply(lambda x: 1 if x else 0)
    # Binarize rater agrees
    df["agree_bin"] = df["rater_agree_label"].apply(lambda x: 1 if str(x).lower().startswith("agree") else 0)

    # Map group -> num_reviewers
    group2num = {g: info["num_reviewers"] for g, info in CSV_URLS.items()}
    df["num_reviewers"] = df["group"].map(group2num)

    # We'll define a helper function
    def majority_label(sub, num_reviewers):
        if num_reviewers == 3:
            n_agree = sub["agree_bin"].sum()
            return 1 if n_agree >= 2 else 0
        elif num_reviewers == 2:
            # pick the first row's label
            return int(sub.iloc[0]["agree_bin"])
        else:
            return 0

    # Now group by (group,case_index,field_index)
    records = []
    gb = df.groupby(["group","case_index","field_index"])
    for key, subdf in gb:
        g, c, f = key
        nr = subdf["num_reviewers"].iloc[0]
        gt = majority_label(subdf, nr)
        llm_label = subdf.iloc[0]["LLM_label"]  # same for all rows in subdf
        records.append({
            "group": g,
            "case_index": c,
            "field_index": f,
            "ground_truth": gt,
            "llm_label": llm_label
        })

    df_conf = pd.DataFrame(records)
    df_conf["TP"] = ((df_conf["llm_label"]==1) & (df_conf["ground_truth"]==1)).astype(int)
    df_conf["FP"] = ((df_conf["llm_label"]==1) & (df_conf["ground_truth"]==0)).astype(int)
    df_conf["TN"] = ((df_conf["llm_label"]==0) & (df_conf["ground_truth"]==0)).astype(int)
    df_conf["FN"] = ((df_conf["llm_label"]==0) & (df_conf["ground_truth"]==1)).astype(int)

    sums = df_conf[["TP","FP","TN","FN"]].sum()
    TP,FP,TN,FN = sums.TP, sums.FP, sums.TN, sums.FN

    accuracy = (TP+TN)/(TP+TN+FP+FN+1e-9)
    precision = TP/(TP+FP+1e-9)
    recall = TP/(TP+FN+1e-9)
    f1 = 2*(precision*recall)/(precision+recall+1e-9)

    metrics = {
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "df_conf_detail": df_conf  # returning the detail if needed
    }
    return metrics


def compute_time_analysis(df_time_all):
    """
    Convert total_time_for_form to numeric if possible, sum across forms per rater,
    compute time_per_case, and then compute mean+95% CI across all raters.
    Returns (df_time_rater, summary_dict)
    """
    # Attempt naive float parse
    def try_float(x):
        try:
            return float(str(x).strip().replace(",","").split()[0])  # e.g. "5 minutes" -> 5.0
        except:
            return np.nan

    df_time = df_time_all.copy()
    df_time["time_num"] = df_time["total_time_for_form"].apply(try_float)

    # Group by rater
    grouped = df_time.groupby("rater_email").agg({
        "time_num": "sum",
        "num_cases_in_form": "sum"
    }).reset_index()

    grouped["time_per_case"] = grouped["time_num"] / grouped["num_cases_in_form"]
    times = grouped["time_per_case"].dropna().values
    mean_time = times.mean()
    std_time = times.std(ddof=1)
    n = len(times)

    if n > 1:
        import math
        sem = std_time / math.sqrt(n)
        ci_lower = mean_time - 1.96*sem
        ci_upper = mean_time + 1.96*sem
    else:
        # If only one rater or no valid times, can't compute meaningful CI
        ci_lower, ci_upper = mean_time, mean_time

    summary = {
        "mean_time_per_case": mean_time,
        "std_time_per_case": std_time,
        "n_raters": n,
        "95%CI_lower": ci_lower,
        "95%CI_upper": ci_upper
    }
    return grouped, summary
