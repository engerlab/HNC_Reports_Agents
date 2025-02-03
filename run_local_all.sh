#!/bin/bash

# Bash script to run the summarizer with local Ollama models
# while showing:
# 1) A spinning 'Processing...' sign, AND
# 2) x/y (Pct%) for pathology_reports, consultation_notes, and treatment_plan_outcomepred
#
# This does NOT change the Python code. We parse the logs to see progress.

INPUT_DIR="/Data/Yujing/HNC_OutcomePred/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp2"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports4.py"

# Temporary log file for Python output
LOGFILE="/tmp/summarizer_progress.log"
rm -f "$LOGFILE"

# 1) Count how many .txt files for each subfolder
PAT_TOTAL=$(find "$INPUT_DIR/pathology_reports" -type f -name '*.txt' 2>/dev/null | wc -l)
CON_TOTAL=$(find "$INPUT_DIR/consultation_notes" -type f -name '*.txt' 2>/dev/null | wc -l)
TP_TOTAL=$(find "$INPUT_DIR/treatment_plan_outcomepred" -type f -name '*.txt' 2>/dev/null | wc -l)

echo "Pathology Reports: $PAT_TOTAL .txt files"
echo "Consultation Notes: $CON_TOTAL .txt files"
echo "Treatment Plan Outcome Predictions: $TP_TOTAL .txt files"
if [ "$PAT_TOTAL" -eq 0 ] && [ "$CON_TOTAL" -eq 0 ] && [ "$TP_TOTAL" -eq 0 ]; then
  echo "No .txt files found in pathology_reports, consultation_notes, or treatment_plan_outcomepred. Exiting."
  exit 1
fi

echo "Starting Summarizer with local Ollama..."

# 2) Run Python in background, capturing logs
python "$PYTHON_SCRIPT" \
  --prompts_dir "$PROMPTS_DIR" \
  --model_type "$MODEL_TYPE" \
  --temperature "$TEMPERATURE" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --embedding_model "$EMBEDDING_MODEL" \
  >"$LOGFILE" 2>&1 &

PID=$!

# 3) Define a spinner
spin='-\|/'
i=0

# 4) While python is running, show spinner + subfolder progress
while kill -0 "$PID" 2>/dev/null; do
  i=$(( (i+1) %4 ))
  spinChar=${spin:$i:1}
  PAT_PROCESSED=$(grep -c "Processing file: .* in folder: pathology_reports" "$LOGFILE")
  CON_PROCESSED=$(grep -c "Processing file: .* in folder: consultation_notes" "$LOGFILE")
  TP_PROCESSED=$(grep -c "Processing file: .* in folder: treatment_plan_outcomepred" "$LOGFILE")
  if [ "$PAT_TOTAL" -gt 0 ]; then
    PAT_PCT=$(( 100 * PAT_PROCESSED / PAT_TOTAL ))
  else
    PAT_PCT=0
  fi
  if [ "$CON_TOTAL" -gt 0 ]; then
    CON_PCT=$(( 100 * CON_PROCESSED / CON_TOTAL ))
  else
    CON_PCT=0
  fi
  if [ "$TP_TOTAL" -gt 0 ]; then
    TP_PCT=$(( 100 * TP_PROCESSED / TP_TOTAL ))
  else
    TP_PCT=0
  fi
  echo -ne "\r[$spinChar] Path: $PAT_PROCESSED/$PAT_TOTAL ($PAT_PCT%)  Cons: $CON_PROCESSED/$CON_TOTAL ($CON_PCT%)  TP: $TP_PROCESSED/$TP_TOTAL ($TP_PCT%)"
  sleep 1
done

# 5) Once the Python finishes, do one final read
PAT_PROCESSED=$(grep -c "Processing file: .* in folder: pathology_reports" "$LOGFILE")
CON_PROCESSED=$(grep -c "Processing file: .* in folder: consultation_notes" "$LOGFILE")
TP_PROCESSED=$(grep -c "Processing file: .* in folder: treatment_plan_outcomepred" "$LOGFILE")
if [ "$PAT_TOTAL" -gt 0 ]; then
  PAT_PCT=$(( 100 * PAT_PROCESSED / PAT_TOTAL ))
else
  PAT_PCT=0
fi
if [ "$CON_TOTAL" -gt 0 ]; then
  CON_PCT=$(( 100 * CON_PROCESSED / CON_TOTAL ))
else
  CON_PCT=0
fi
if [ "$TP_TOTAL" -gt 0 ]; then
  TP_PCT=$(( 100 * TP_PROCESSED / TP_TOTAL ))
else
  TP_PCT=0
fi
echo -e "\r[+] Path: $PAT_PROCESSED/$PAT_TOTAL ($PAT_PCT%), Cons: $CON_PROCESSED/$CON_TOTAL ($CON_PCT%), TP: $TP_PROCESSED/$TP_TOTAL ($TP_PCT%). Done!"

# (Optional) Uncomment the next line to show final logs:
# echo "Logs are in $LOGFILE"
