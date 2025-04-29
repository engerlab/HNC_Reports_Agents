#!/usr/bin/env python3
# fetch_gsheet_gspread.py

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# ----------------------------------------------------------------------------
# 1. PATH to your downloaded service account JSON key file
# ----------------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "/Data/Yujing/HNC_OutcomePred/Google_Cloud/HNC_LLM/hnc-llm-ffbe7524baff.json"

# ----------------------------------------------------------------------------
# 2. SCOPES: Which APIs your service account will use.
#    For read-only on Sheets, you can use only the Sheets scope:
#    "https://www.googleapis.com/auth/spreadsheets.readonly"
#    If you need to read/write, add Drive scope, etc.
# ----------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"  # optional if you need drive access
]

# ----------------------------------------------------------------------------
# 3. Your Google Sheet info:
#    - SHEET_ID is the long ID from the Google Sheets URL
#    - GID is the numeric tab ID if you want a specific worksheet
# ----------------------------------------------------------------------------
# Raw url example: https://docs.google.com/spreadsheets/d/1PuzDr0uRjfeNDOgQSbNMq4FJ6A3kRt0rJJDeI0GRhlo/edit?gid=1217539876#gid=1217539876 
SHEET_ID = "1PuzDr0uRjfeNDOgQSbNMq4FJ6A3kRt0rJJDeI0GRhlo"
GID = 1217539876  # "Form Responses" tab or whichever tab you want

# ----------------------------------------------------------------------------
# 4. OUTPUT FILE LOCATION
# ----------------------------------------------------------------------------
OUTPUT_CSV_PATH = (
    "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/"
    "Clinician_Evaluation/tmp/Group1_PathOnly_Clinician_eval_googleapi.csv"
)

def main():
    # ------------------------------------------------------------------------
    # A) AUTHENTICATE VIA SERVICE ACCOUNT
    # ------------------------------------------------------------------------
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=SCOPES
    )
    gc = gspread.authorize(credentials)

    # ------------------------------------------------------------------------
    # B) OPEN SHEET AND SELECT WORKSHEET
    # ------------------------------------------------------------------------
    # Option 1: Open by SHEET_ID, then get the correct worksheet by GID
    sh = gc.open_by_key(SHEET_ID)
    worksheet = None

    # gspread provides get_worksheet(index) by 0-based index
    # or get_worksheet_by_id(gid) in newer versions of gspread.
    try:
        # If you have a recent version of gspread:
        #   worksheet = sh.get_worksheet_by_id(GID)
        # Otherwise, you might do something like
        #   all_worksheets = sh.worksheets()
        #   print([ (ws.title, ws.id) for ws in all_worksheets ])
        #   Then pick the matching one by .id or .title.
        worksheet = sh.get_worksheet_by_id(GID)
    except AttributeError:
        # Fallback if the installed gspread doesn't have get_worksheet_by_id
        # then you can match by index or by title:
        # e.g. worksheet = sh.worksheet("Form Responses 1")
        raise RuntimeError(
            "Your gspread version doesn’t support get_worksheet_by_id. "
            "Upgrade via pip install --upgrade gspread, or use sh.worksheet('TabName')."
        )

    # ------------------------------------------------------------------------
    # C) FETCH ALL RECORDS INTO A PANDAS DATAFRAME
    # ------------------------------------------------------------------------
    # get_all_records() returns a list of dictionaries
    rows = worksheet.get_all_records()
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # D) BASIC ANALYSIS AND OUTPUT
    # ------------------------------------------------------------------------
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print(f"\n--- Number of Rows: {len(df)} ---")
    print(f"--- Number of Columns: {len(df.columns)} ---")

    print("\n--- DataFrame Info ---")
    print(df.info())

    # Save as CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n--- DataFrame Saved as {OUTPUT_CSV_PATH} ---")


if __name__ == "__main__":
    main()
