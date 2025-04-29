# main_analyze.py
import os
import pandas as pd

from config import CSV_URLS, RESULTS_FOLDER
from fetch_and_parse import fetch_and_parse_all
from compute_metrics import (
    compute_fleiss_kappa_3rater,
    compute_confusion_matrix,
    compute_time_analysis,
)

def main():
    # 1) Ensure results folder exists
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # 2) Fetch & parse all data
    df_parsed_all, df_time_all = fetch_and_parse_all(CSV_URLS)
    # Save immediate parsed data for debugging if desired
    df_parsed_all.to_csv(os.path.join(RESULTS_FOLDER, "parsed_all_long.csv"), index=False)
    df_time_all.to_csv(os.path.join(RESULTS_FOLDER, "parsed_all_time.csv"), index=False)

    # 3) Compute Fleiss’ Kappa (for 3-rater groups only)
    overall_kappa_3 = compute_fleiss_kappa_3rater(df_parsed_all, CSV_URLS)
    print(f"Overall Fleiss’ Kappa (3-rater groups): {overall_kappa_3:.3f}")

    # 4) Compute confusion matrix & derived metrics
    metrics = compute_confusion_matrix(df_parsed_all, CSV_URLS)
    print("\n=== Confusion Matrix Totals (all fields, all groups) ===")
    print(f"TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
    print(f"Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    # Optionally, save the detail confusion DataFrame:
    df_conf_detail = metrics["df_conf_detail"]
    df_conf_detail.to_csv(os.path.join(RESULTS_FOLDER, "confusion_detail.csv"), index=False)

    # 5) Extract Comments & Ratings
    df_comments_ratings = df_parsed_all[[
        "group", "case_index", "rater_email", "comment_for_case", "rating_for_case"
    ]].drop_duplicates()
    df_comments_ratings.to_csv(os.path.join(RESULTS_FOLDER, "comments_and_ratings.csv"), index=False)

    # 6) Analyze time
    df_time_rater, time_summary = compute_time_analysis(df_time_all)
    df_time_rater.to_csv(os.path.join(RESULTS_FOLDER, "rater_time_analysis.csv"), index=False)

    print("\n=== Time Analysis ===")
    print(df_time_rater)
    print(f"\nOverall mean time_per_case = {time_summary['mean_time_per_case']:.2f} "
          f"({time_summary['95%CI_lower']:.2f} - {time_summary['95%CI_upper']:.2f} 95% CI), "
          f"n={time_summary['n_raters']}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
