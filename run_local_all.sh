#!/bin/bash

# Usage: ./run_local_all.sh [report_type]
# report_type can be: pathology_reports, consultation_notes, treatment_plan_outcomepred, or "all" (default).

INPUT_DIR="/media/yujing/One Touch3/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp3"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
LOCAL_MODEL="llama3.3:latest"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports5.py"

if [ -z "$1" ] || [ "$1" == "all" ]; then
  REPORT_TYPE="all"
else
  REPORT_TYPE="$1"
fi

# For folder naming:
# - We use "PathologyReports" for pathology_reports
# - "ConsultRedacted" for consultation_notes
# - "PathConsCombined" for treatment_plan_outcomepred

# Count files for each type
PAT_TOTAL=0
CON_TOTAL=0
TP_TOTAL=0

if [[ "$REPORT_TYPE" == *"pathology_reports"* ]] || [ "$REPORT_TYPE" == "all" ]; then
  PAT_TOTAL=$(find "$INPUT_DIR/PathologyReports" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

if [[ "$REPORT_TYPE" == *"consultation_notes"* ]] || [ "$REPORT_TYPE" == "all" ]; then
  CON_TOTAL=$(find "$INPUT_DIR/ConsultRedacted" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

if [[ "$REPORT_TYPE" == *"treatment_plan_outcomepred"* ]] || [ "$REPORT_TYPE" == "all" ]; then
  TP_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

echo "Pathology Reports (PathologyReports): $PAT_TOTAL .txt files"
echo "Consultation Notes (ConsultRedacted): $CON_TOTAL .txt files"
echo "Treatment Plan (PathConsCombined): $TP_TOTAL .txt files"

if [ "$PAT_TOTAL" -eq 0 ] && [ "$CON_TOTAL" -eq 0 ] && [ "$TP_TOTAL" -eq 0 ]; then
  echo "No .txt files found in the named subfolders. Exiting."
  exit 1
fi

echo "Starting Summarizer with local Ollama..."

LOGFILE="/tmp/summarizer_progress.log"
rm -f "$LOGFILE"

python "$PYTHON_SCRIPT" \
  --prompts_dir "$PROMPTS_DIR" \
  --model_type "$MODEL_TYPE" \
  --temperature "$TEMPERATURE" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --embedding_model "$EMBEDDING_MODEL" \
  --report_type "$REPORT_TYPE" \
  --local_model "$LOCAL_MODEL" \
  >"$LOGFILE" 2>&1 &

PID=$!

# Spinner to show progress
spin='-\|/'
i=0

while kill -0 "$PID" 2>/dev/null; do
  i=$(( (i+1) %4 ))
  spinChar=${spin:$i:1}
  
  # Count processed files:
  PAT_PROCESSED=$(grep -c "Processing file: .* in folder: PathologyReports" "$LOGFILE")
  CON_PROCESSED=$(grep -c "Processing file: .* in folder: ConsultRedacted" "$LOGFILE")
  TP_PROCESSED=$(grep -c "Processing combined file: .* in folder: PathConsCombined" "$LOGFILE")

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

  echo -ne "\r[$spinChar] Path: $PAT_PROCESSED/$PAT_TOTAL (${PAT_PCT}%)  Cons: $CON_PROCESSED/$CON_TOTAL (${CON_PCT}%)  TP: $TP_PROCESSED/$TP_TOTAL (${TP_PCT}%)"
  sleep 1
done

# Final read
PAT_PROCESSED=$(grep -c "Processing file: .* in folder: PathologyReports" "$LOGFILE")
CON_PROCESSED=$(grep -c "Processing file: .* in folder: ConsultRedacted" "$LOGFILE")
TP_PROCESSED=$(grep -c "Processing combined file: .* in folder: PathConsCombined" "$LOGFILE")

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
echo "Logs are in $LOGFILE"
