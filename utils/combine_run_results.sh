#!/bin/bash
# combine_run_results.sh (revised)

###############################################################################
# 1) Source & Destination Paths
###############################################################################
SOURCE_EXP13="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13"
SOURCE_EXP38="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt38"
TOKEN_COUNTS_CSV="/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode2.csv"

DEST_EXP14="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14"

# Make sure the destination folder exists
mkdir -p "$DEST_EXP14"

###############################################################################
# 2) Copy supporting CSV files
###############################################################################
echo "Copying CSV files..."

# 2a) filtered_cases_mode2.csv from Exp38 => Exp14
if [ -f "$SOURCE_EXP38/filtered_cases_mode2.csv" ]; then
  cp "$SOURCE_EXP38/filtered_cases_mode2.csv" "$DEST_EXP14/filtered_cases_mode2.csv"
  echo "Copied filtered_cases_mode2.csv => $DEST_EXP14"
else
  echo "Warning: filtered_cases_mode2.csv not found in $SOURCE_EXP38"
fi

# 2b) precompute_tokens_mode2.csv from your Token_Counts folder => Exp14
if [ -f "$TOKEN_COUNTS_CSV" ]; then
  cp "$TOKEN_COUNTS_CSV" "$DEST_EXP14/precompute_tokens_mode2.csv"
  echo "Copied precompute_tokens_mode2.csv => $DEST_EXP14"
else
  echo "Warning: precompute_tokens_mode2.csv not found at $TOKEN_COUNTS_CSV"
fi

# 2c) processing_times.csv from each run, renamed
if [ -f "$SOURCE_EXP13/processing_times.csv" ]; then
  cp "$SOURCE_EXP13/processing_times.csv" "$DEST_EXP14/processing_times_exp13.csv"
  echo "Copied processing_times.csv => processing_times_exp13.csv"
fi

if [ -f "$SOURCE_EXP38/processing_times.csv" ]; then
  cp "$SOURCE_EXP38/processing_times.csv" "$DEST_EXP14/processing_times_exp38.csv"
  echo "Copied processing_times.csv => processing_times_exp38.csv"
fi

###############################################################################
# 3) Copy text_summaries & embeddings from Exp13
#    (They were stored in path_consult_reports/<case_id>)
#    We place them into Exp14/text_summaries/mode2_combined_twostep/<case_id>
###############################################################################
echo "Copying subfolders from Exp13 -> Exp14..."

mkdir -p "$DEST_EXP14/text_summaries/mode2_combined_twostep"
mkdir -p "$DEST_EXP14/embeddings/mode2_combined_twostep"

# From Exp13, the final summaries live in text_summaries/path_consult_reports/<case_id>
# So we copy the *contents* of that folder into the new 'mode2_combined_twostep' location.
if [ -d "$SOURCE_EXP13/text_summaries/path_consult_reports" ]; then
  rsync -av --exclude='structured_data*' \
    "$SOURCE_EXP13/text_summaries/path_consult_reports/" \
    "$DEST_EXP14/text_summaries/mode2_combined_twostep/"
else
  echo "Warning: No path_consult_reports subfolder found under Exp13 text_summaries."
fi

# Similarly for embeddings => embeddings/path_consult_reports/<case_id>
if [ -d "$SOURCE_EXP13/embeddings/path_consult_reports" ]; then
  rsync -av --exclude='structured_data*' \
    "$SOURCE_EXP13/embeddings/path_consult_reports/" \
    "$DEST_EXP14/embeddings/mode2_combined_twostep/"
else
  echo "Warning: No path_consult_reports subfolder found under Exp13 embeddings."
fi

###############################################################################
# 4) Copy text_summaries & embeddings from Exp38 (overwriting duplicates)
#    In Exp38, they're already in text_summaries/mode2_combined_twostep/<case_id>
###############################################################################
echo "Copying subfolders from Exp38 -> Exp14 (overwriting duplicates if any)..."

# The folder names in Exp38 are already 'mode2_combined_twostep'
if [ -d "$SOURCE_EXP38/text_summaries/mode2_combined_twostep" ]; then
  rsync -av --exclude='structured_data*' \
    "$SOURCE_EXP38/text_summaries/mode2_combined_twostep/" \
    "$DEST_EXP14/text_summaries/mode2_combined_twostep/"
fi

if [ -d "$SOURCE_EXP38/embeddings/mode2_combined_twostep" ]; then
  rsync -av --exclude='structured_data*' \
    "$SOURCE_EXP38/embeddings/mode2_combined_twostep/" \
    "$DEST_EXP14/embeddings/mode2_combined_twostep/"
fi

###############################################################################
# 5) Cleanup any extraneous subfolders (if needed)
###############################################################################
find "$DEST_EXP14" -type d -name "structured_data*" -exec rm -rf {} + 2>/dev/null

echo "Merged text_summaries & embeddings. Now building a final merged CSV of processing_times..."

###############################################################################
# 6) Merge processing_times_exp13.csv & processing_times_exp38.csv 
#    => final_merged_processing_times.csv
#    Prefer Exp38 rows if 'file' is duplicated, else use Exp13.
#    Keep only: file, report_type, process_time_ms
###############################################################################
python3 <<EOF
import pandas as pd
import os

csv13 = os.path.join("$DEST_EXP14", "processing_times_exp13.csv")
csv38 = os.path.join("$DEST_EXP14", "processing_times_exp38.csv")
out_csv = os.path.join("$DEST_EXP14", "final_merged_processing_times.csv")

# If neither CSV is present, skip
if not os.path.isfile(csv13) and not os.path.isfile(csv38):
    print("No processing_times from either run, skipping merge.")
    raise SystemExit(0)

# Load data
df13 = pd.DataFrame()
if os.path.isfile(csv13):
    df13 = pd.read_csv(csv13)

df38 = pd.DataFrame()
if os.path.isfile(csv38):
    df38 = pd.read_csv(csv38)

# unify columns
all_cols = set(df13.columns.tolist()) | set(df38.columns.tolist())
df13 = df13.reindex(columns=all_cols, fill_value="")
df38 = df38.reindex(columns=all_cols, fill_value="")

# Convert 'file' col to string if present
if 'file' in df13.columns:
    df13['file'] = df13['file'].astype(str)
if 'file' in df38.columns:
    df38['file'] = df38['file'].astype(str)

# Build dict keyed by 'file'
d13 = {}
if 'file' in df13.columns:
    for _, row in df13.iterrows():
        d13[row['file']] = row

d38 = {}
if 'file' in df38.columns:
    for _, row in df38.iterrows():
        d38[row['file']] = row

all_files = set(d13.keys()) | set(d38.keys())
final_rows = []
for f in all_files:
    if f in d38:  # prefer Exp38 row if present
        final_rows.append(d38[f])
    else:
        final_rows.append(d13[f])

df_final = pd.DataFrame(final_rows)

# keep only these columns if they exist
desired_cols = ['file','report_type','process_time_ms']
found_cols = [c for c in desired_cols if c in df_final.columns]
df_final = df_final[found_cols]

df_final.to_csv(out_csv, index=False)
print(f"Merged processing_times => {out_csv} with {len(all_files)} unique 'file' entries.")
print(f"Kept columns: {found_cols}")
EOF

echo "All done! Final merged folder is => $DEST_EXP14"
ls -R "$DEST_EXP14"

