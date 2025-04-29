# fetch_and_parse.py
import pandas as pd

def parse_group_data(group_name, df, num_cases, num_reviewers):
    """
    Parses the raw DataFrame for a single group's CSV into a structured format.
    Returns two DataFrames:
      - df_parsed: Long-form with rows = (group, case_index, rater_email, field_index,
                                         field_name, llm_extracted_value, llm_value_inferred,
                                         rater_agree_label, comment_for_case, rating_for_case)
      - df_time:   Per-rater time records for how long it took to finish that entire form
    """
    records = []
    time_records = []

    START_COL = 2  # columns 0=Timestamp, 1=Email
    CASE_BLOCK_SIZE = 32
    TIME_COL_IDX = START_COL + CASE_BLOCK_SIZE * num_cases  # last col for total time

    all_cols = df.columns.tolist()

    for row_i in range(len(df)):
        row_data = df.iloc[row_i]
        rater_email = row_data["Email Address"]

        # Grab total time from the last column
        total_time = row_data.iloc[TIME_COL_IDX]

        # Record time
        time_records.append({
            "group": group_name,
            "rater_email": rater_email,
            "total_time_for_form": total_time,
            "num_cases_in_form": num_cases
        })

        # Parse each case
        for case_idx in range(num_cases):
            cstart = START_COL + case_idx * CASE_BLOCK_SIZE
            cend = cstart + CASE_BLOCK_SIZE

            # First 30 columns => "Agree/Disagree" fields
            # 31st => "Optional Comment"
            # 32nd => "LLM rating"
            agree_cols = all_cols[cstart : cstart+30]
            comment_col = all_cols[cstart+30]
            rating_col = all_cols[cstart+31]

            comment_val = row_data[comment_col]
            rating_val = row_data[rating_col]

            # Iterate over the 30 extracted-field columns
            for field_i, col_name in enumerate(agree_cols):
                rater_agree = row_data[col_name]  # "Agree", "Disagree", "Unsure", etc.

                # Attempt to parse the bracket portion: e.g. "[Sex: M]"
                bracket_content = ""
                if "[" in col_name and "]" in col_name:
                    bracket_content = col_name.split("[", 1)[1].rsplit("]", 1)[0]
                
                if ": " in bracket_content:
                    parts = bracket_content.split(": ", 1)
                    extracted_fieldname = parts[0].strip()
                    extracted_value = parts[1].strip()
                else:
                    extracted_fieldname = bracket_content
                    extracted_value = ""

                llm_value_inferred = (extracted_value.lower() != "not inferred")

                records.append({
                    "group": group_name,
                    "case_index": case_idx,
                    "rater_email": rater_email,
                    "field_index": field_i,
                    "field_name": extracted_fieldname,
                    "llm_extracted_value": extracted_value,
                    "llm_value_inferred": llm_value_inferred,
                    "rater_agree_label": rater_agree,
                    "comment_for_case": comment_val,
                    "rating_for_case": rating_val
                })

    df_parsed = pd.DataFrame(records)
    df_time = pd.DataFrame(time_records)
    return df_parsed, df_time


def fetch_and_parse_all(CSV_URLS):
    """
    Given the dictionary CSV_URLS, read each URL into pandas,
    parse them, and return combined dataframes for parsed content and time.
    """
    all_parsed = []
    all_time = []

    for group_name, info in CSV_URLS.items():
        url = info["url"]
        num_cases = info["num_cases"]
        num_reviewers = info["num_reviewers"]

        df_raw = pd.read_csv(url)
        df_parsed, df_time = parse_group_data(group_name, df_raw, num_cases, num_reviewers)

        all_parsed.append(df_parsed)
        all_time.append(df_time)

    df_parsed_all = pd.concat(all_parsed, ignore_index=True)
    df_time_all = pd.concat(all_time, ignore_index=True)

    return df_parsed_all, df_time_all
