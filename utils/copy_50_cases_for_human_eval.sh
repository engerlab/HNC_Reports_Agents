#!/bin/bash
#
# copy_50_cases_for_human_eval.sh
#
# Purpose:
#   From the final_50_cutoff_selection.csv (50 selected patient IDs),
#   copy only those 50 subfolders + partial CSVs into a new destination,
#   and also copy the final_50_cutoff_selection.csv itself there.
#
###############################################################################

set -e  # exit on error

# 1) Paths
CSV_50="/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/CaseSelection/Exp14/final_50_cutoff_selection.csv"
SOURCE_EXP14="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp14"
DEST="/Data/Yujing/HNC_OutcomePred/Reports_Agents/Experiments/CaseSelected_HumanEval"

# 2) Make sure the destination folder structure exists
mkdir -p "$DEST/text_summaries/mode2_combined_twostep"
mkdir -p "$DEST/embeddings/mode2_combined_twostep"

# Also copy the final_50_cutoff_selection.csv into the new destination for reference
cp "$CSV_50" "$DEST/final_50_cutoff_selection.csv"

# 3) Extract the 50 patient IDs from the CSV
#    (assumes 'patient_id' is a header in the first column)
patient_ids=$(tail -n +2 "$CSV_50" | cut -d, -f1)

# 4) Copy subfolders for each of the 50 case IDs
for pid in $patient_ids; do
  # Text summaries
  src_text="$SOURCE_EXP14/text_summaries/mode2_combined_twostep/$pid"
  dest_text="$DEST/text_summaries/mode2_combined_twostep"
  if [ -d "$src_text" ]; then
    rsync -av "$src_text" "$dest_text/"
  else
    echo "Warning: text_summaries for patient '$pid' not found in Exp14."
  fi

  # Embeddings
  src_emb="$SOURCE_EXP14/embeddings/mode2_combined_twostep/$pid"
  dest_emb="$DEST/embeddings/mode2_combined_twostep"
  if [ -d "$src_emb" ]; then
    rsync -av "$src_emb" "$dest_emb/"
  else
    echo "Warning: embeddings for patient '$pid' not found in Exp14."
  fi
done

# 5) Filter the CSVs: final_merged_processing_times.csv & precompute_tokens_mode2.csv
#    Keep only the 50 patient IDs
python3 <<EOF
import pandas as pd
import os

csv_50 = "${CSV_50}"
df_50 = pd.read_csv(csv_50)
pids = set(df_50["patient_id"].astype(str))

# 5a) final_merged_processing_times.csv
fmpt_csv = os.path.join("${SOURCE_EXP14}", "final_merged_processing_times.csv")
if os.path.isfile(fmpt_csv):
    df_fmpt = pd.read_csv(fmpt_csv)
    # 'file' is something like "12345.txt"; strip ".txt" to compare with PID
    df_fmpt["base_pid"] = df_fmpt["file"].str.replace(r"\.txt$", "", regex=True)
    df_filt = df_fmpt[df_fmpt["base_pid"].isin(pids)].drop(columns="base_pid", errors="ignore")
    out_fmpt = os.path.join("${DEST}", "final_merged_processing_times.csv")
    df_filt.to_csv(out_fmpt, index=False)
    print(f"Filtered final_merged_processing_times.csv => {len(df_filt)} rows")
else:
    print("Warning: final_merged_processing_times.csv not found in Exp14.")

# 5b) precompute_tokens_mode2.csv
ptm2_csv = os.path.join("${SOURCE_EXP14}", "precompute_tokens_mode2.csv")
if os.path.isfile(ptm2_csv):
    df_tokens = pd.read_csv(ptm2_csv)
    df_tokens["base_pid"] = df_tokens["file"].str.replace(r"\.txt$", "", regex=True)
    df_tokens_filt = df_tokens[df_tokens["base_pid"].isin(pids)].drop(columns="base_pid", errors="ignore")
    out_tokens = os.path.join("${DEST}", "precompute_tokens_mode2.csv")
    df_tokens_filt.to_csv(out_tokens, index=False)
    print(f"Filtered precompute_tokens_mode2.csv => {len(df_tokens_filt)} rows")
else:
    print("Warning: precompute_tokens_mode2.csv not found in Exp14.")
EOF

echo
echo "All done. Copied subfolders & filtered CSVs for the 50 selected patients,"
echo "and also copied final_50_cutoff_selection.csv."
echo "Destination folder: ${DEST}"
