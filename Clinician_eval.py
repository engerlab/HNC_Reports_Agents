#!/usr/bin/env python3
# fetch_google_sheet.py

import pandas as pd

# ------------------------------------------------------------------
# REPLACE THIS URL with the CSV export link you constructed:
# (the one that looks like "?format=csv&id=...&gid=...")
# ------------------------------------------------------------------
# Raw url example: https://docs.google.com/spreadsheets/d/1PuzDr0uRjfeNDOgQSbNMq4FJ6A3kRt0rJJDeI0GRhlo/edit?gid=1217539876#gid=1217539876 
# The URL should be the one you get from the "Share" option in Google Sheets 
# Need the "export?format=csv&" to be added to the URL 

SPREADSHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1PuzDr0uRjfeNDOgQSbNMq4FJ6A3kRt0rJJDeI0GRhlo/"
    "export?format=csv&"
    "id=1PuzDr0uRjfeNDOgQSbNMq4FJ6A3kRt0rJJDeI0GRhlo&"
    "gid=1217539876"
)

def main():
    # 1. Read data from Google Sheets CSV export link
    df = pd.read_csv(SPREADSHEET_CSV_URL)
    
    # 2. Print the first few rows
    print("\n--- First 5 Rows ---")
    print(df.head())
    print(df.columns)
    print(f"\n--- Number of Rows: {len(df)} ---")
    print(f"--- Number of Columns: {len(df.columns)} ---")
    
    # 3. Basic info/stats about the DataFrame
    print("\n--- DataFrame Info ---")
    print(df.info())

    # save as a CSV file
    df.to_csv("/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Clinician_Evaluation/tmp/Group1_PathOnly_Clinician_eval.csv", index=False)
    print("\n--- DataFrame Saved as Clinician_eval.csv ---")

    
    # print("\n--- Descriptive Statistics ---")
    # print(df.describe(include='all'))
    
if __name__ == "__main__":
    main()
