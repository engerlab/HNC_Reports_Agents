#!/bin/bash

# Bash script to run summarizer with local Ollama models
# and show separate progress for pathology_reports and consultation_notes
# based on how many "Processing file:" lines appear in the Python log.

INPUT_DIR="/Data/Yujing/HNC_OutcomePred/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp1"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports3.py"

LOGFILE="/tmp/summarizer_progress.log"
rm -f "$LOGFILE"

# 1) Count how many .txt files in each subfolder
PAT_TOTAL=$(find "$INPUT_DIR/pathology_reports" -type f -name '*.txt' 2>/dev/null | wc -l)
CON_TOTAL=$(find "$INPUT_DIR/consultation_notes" -type f -name '*.txt' 2>/dev/null | wc -l)

echo "Pathology Reports: $PAT_TOTAL files"
echo "Consultation Notes: $CON_TOTAL files"

# 2) If both are zero, exit
if [ "$PAT_TOTAL" -eq 0 ] && [ "$CON_TOTAL" -eq 0 ]; then
  echo "No .txt files found in either pathology_reports or consultation_notes."
  exit 1
fi

# 3) Run the Python script in the background, capturing logs
python "$PYTHON_SCRIPT" \
  --prompts_dir "$PROMPTS_DIR" \
  --model_type "$MODEL_TYPE" \
  --temperature "$TEMPERATURE" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --embedding_model "$EMBEDDING_MODEL" \
  > "$LOGFILE" 2>&1 &

PID=$!

# 4) Show separate progress until Python finishes
LAST_PAT=0
LAST_CON=0

while kill -0 "$PID" 2>/dev/null; do
  # Count how many lines mention "Processing file: ??? in folder: pathology_reports"
  PAT_PROCESSED=$(grep -c "Processing file: .* in folder: pathology_reports" "$LOGFILE")

  # Count how many lines mention "Processing file: ??? in folder: consultation_notes"
  CON_PROCESSED=$(grep -c "Processing file: .* in folder: consultation_notes" "$LOGFILE")

  # Calculate percentages
  if [ "$PAT_TOTAL" -gt 0 ]; then
    PAT_PERCENT=$(( 100 * PAT_PROCESSED / PAT_TOTAL ))
  else
    PAT_PERCENT=0
  fi

  if [ "$CON_TOTAL" -gt 0 ]; then
    CON_PERCENT=$(( 100 * CON_PROCESSED / CON_TOTAL ))
  else
    CON_PERCENT=0
  fi

  # Only update if there's a change
  if [ "$PAT_PROCESSED" -ne "$LAST_PAT" ] || [ "$CON_PROCESSED" -ne "$LAST_CON" ]; then
    echo -ne "\r$PAT_PROCESSED/$PAT_TOTAL ($PAT_PERCENT%) pathology reports processed, \
$CON_PROCESSED/$CON_TOTAL ($CON_PERCENT%) consultation notes processed"
    LAST_PAT=$PAT_PROCESSED
    LAST_CON=$CON_PROCESSED
  fi

  sleep 1
done

# 5) After Python finishes, do a final update
PAT_PROCESSED=$(grep -c "Processing file: .* in folder: pathology_reports" "$LOGFILE")
CON_PROCESSED=$(grep -c "Processing file: .* in folder: consultation_notes" "$LOGFILE")
if [ "$PAT_TOTAL" -gt 0 ]; then
  PAT_PERCENT=$(( 100 * PAT_PROCESSED / PAT_TOTAL ))
else
  PAT_PERCENT=0
fi
if [ "$CON_TOTAL" -gt 0 ]; then
  CON_PERCENT=$(( 100 * CON_PROCESSED / CON_TOTAL ))
else
  CON_PERCENT=0
fi

echo -e "\r$PAT_PROCESSED/$PAT_TOTAL ($PAT_PERCENT%) pathology reports processed, \
$CON_PROCESSED/$CON_TOTAL ($CON_PERCENT%) consultation notes processed. Done!"
echo
echo "Script finished. Logs in $LOGFILE."
