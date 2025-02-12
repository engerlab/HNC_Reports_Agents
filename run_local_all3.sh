#!/bin/bash

# Usage: ./run_local_all2.sh [report_type]
# report_type can be a comma-separated list of:
#   pathology_reports,
#   consultation_notes,
#   treatment_plan_outcomepred,
#   path_consult_reports,
#   cot_treatment_plan_outcomepred,
#   or "all" (default).
#
# Example:
#   bash /Data/Yujing/HNC_OutcomePred/Reports_Agents/run_local_all2.sh "path_consult_reports"

#############################
# 1) Configuration
#############################
INPUT_DIR="/media/yujing/One Touch3/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp12"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
LOCAL_MODEL="llama3.3:latest"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/hnc_reports_agent2.py"

#############################
# 2) Trap for Process Termination: killing all child processes in Python script as well. 
#############################
trap "echo 'Terminating all processes...'; kill -- -$$; exit 1" SIGINT SIGTERM

#############################
# 3) Parse Command-Line Argument
#############################
if [ -z "$1" ] || [ "$1" == "all" ]; then
  REPORT_TYPE="all"
else
  REPORT_TYPE="$1"
fi

# Convert to lowercase
REPORT_TYPE="${REPORT_TYPE,,}"

# Split on commas into an array
IFS=',' read -ra RT_ARRAY <<< "$REPORT_TYPE"
CLEAN_TYPES=()
for elem in "${RT_ARRAY[@]}"; do
  CLEAN_TYPES+=( "$(echo -n "$elem" | xargs)" )
done

# Initialize booleans for each recognized type
IS_PATHOLOGY=false
IS_CONSULT=false
IS_TP=false   # old treatment_plan_outcomepred
IS_PC=false   # path_consult_reports
IS_COT=false  # cot_treatment_plan_outcomepred

for t in "${CLEAN_TYPES[@]}"; do
  case "$t" in
    "pathology_reports")
      IS_PATHOLOGY=true
      ;;
    "consultation_notes")
      IS_CONSULT=true
      ;;
    "treatment_plan_outcomepred")
      IS_TP=true
      ;;
    "path_consult_reports")
      IS_PC=true
      ;;
    "cot_treatment_plan_outcomepred")
      IS_COT=true
      ;;
    "all")
      IS_PATHOLOGY=true
      IS_CONSULT=true
      IS_TP=true
      IS_PC=true
      IS_COT=true
      ;;
    *)
      echo "Warning: unrecognized report type '$t'"
      ;;
  esac
done

#############################
# 4) Count .txt Files for Selected Types
#############################
PAT_TOTAL=0
CON_TOTAL=0
TP_TOTAL=0
PC_TOTAL=0
COT_TOTAL=0

if [ "$IS_PATHOLOGY" = true ]; then
  PAT_TOTAL=$(find "$INPUT_DIR/PathologyReports" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

if [ "$IS_CONSULT" = true ]; then
  CON_TOTAL=$(find "$INPUT_DIR/ConsultRedacted" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

if [ "$IS_TP" = true ]; then
  TP_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

if [ "$IS_PC" = true ]; then
  PC_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

if [ "$IS_COT" = true ]; then
  COT_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
fi

echo "File counts based on selection:"
[ "$IS_PATHOLOGY" = true ] && echo "  Pathology Reports (PathologyReports): $PAT_TOTAL .txt files"
[ "$IS_CONSULT" = true ] && echo "  Consultation Notes (ConsultRedacted): $CON_TOTAL .txt files"
[ "$IS_TP" = true ] && echo "  Treatment Plan (old) (PathConsCombined): $TP_TOTAL .txt files"
[ "$IS_PC" = true ] && echo "  Combined Path+Consult Extraction (PathConsCombined): $PC_TOTAL .txt files"
[ "$IS_COT" = true ] && echo "  CoT Treatment Plan (PathConsCombined): $COT_TOTAL .txt files"

if [ "$PAT_TOTAL" -eq 0 ] && [ "$CON_TOTAL" -eq 0 ] && [ "$TP_TOTAL" -eq 0 ] && [ "$PC_TOTAL" -eq 0 ] && [ "$COT_TOTAL" -eq 0 ]; then
  echo "No .txt files found in the expected subfolders. Exiting."
  exit 1
fi

echo "Starting Summarizer with local Ollama..."

#############################
# 5) Run Python Script in Background
#############################
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

#############################
# 6) Spinner and Dynamic Progress
#############################
spin='-\|/'
i=0
while kill -0 "$PID" 2>/dev/null; do
  i=$(( (i+1) % 4 ))
  spinChar=${spin:$i:1}

  PAT_PROCESSED=$(grep -c "Processing file: .* in folder: PathologyReports" "$LOGFILE")
  CON_PROCESSED=$(grep -c "Processing file: .* in folder: ConsultRedacted" "$LOGFILE")
  TP_PROCESSED=$(grep -c "Processing combined file: .* in folder: PathConsCombined" "$LOGFILE")
  PC_PROCESSED=$(grep -c "Processing combined path+consult file:" "$LOGFILE")
  COT_PROCESSED=$(grep -c "Processing CoT-based plan:" "$LOGFILE")

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
  if [ "$PC_TOTAL" -gt 0 ]; then
    PC_PCT=$(( 100 * PC_PROCESSED / PC_TOTAL ))
  else
    PC_PCT=0
  fi
  if [ "$COT_TOTAL" -gt 0 ]; then
    COT_PCT=$(( 100 * COT_PROCESSED / COT_TOTAL ))
  else
    COT_PCT=0
  fi

  progress_str=""
  [ "$IS_PATHOLOGY" = true ] && progress_str+="Path: $PAT_PROCESSED/$PAT_TOTAL (${PAT_PCT}%)  "
  [ "$IS_CONSULT" = true ] && progress_str+="Cons: $CON_PROCESSED/$CON_TOTAL (${CON_PCT}%)  "
  [ "$IS_TP" = true ] && progress_str+="OldTP: $TP_PROCESSED/$TP_TOTAL (${TP_PCT}%)  "
  [ "$IS_PC" = true ] && progress_str+="PC: $PC_PROCESSED/$PC_TOTAL (${PC_PCT}%)  "
  [ "$IS_COT" = true ] && progress_str+="CoT: $COT_PROCESSED/$COT_TOTAL (${COT_PCT}%)"
  
  echo -ne "\r[$spinChar] $progress_str"
  sleep 1
done

#############################
# 7) Final Progress Report
#############################
PAT_PROCESSED=$(grep -c "Processing file: .* in folder: PathologyReports" "$LOGFILE")
CON_PROCESSED=$(grep -c "Processing file: .* in folder: ConsultRedacted" "$LOGFILE")
TP_PROCESSED=$(grep -c "Processing combined file: .* in folder: PathConsCombined" "$LOGFILE")
PC_PROCESSED=$(grep -c "Processing combined path+consult file:" "$LOGFILE")
COT_PROCESSED=$(grep -c "Processing CoT-based plan:" "$LOGFILE")

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
if [ "$PC_TOTAL" -gt 0 ]; then
  PC_PCT=$(( 100 * PC_PROCESSED / PC_TOTAL ))
else
  PC_PCT=0
fi
if [ "$COT_TOTAL" -gt 0 ]; then
  COT_PCT=$(( 100 * COT_PROCESSED / COT_TOTAL ))
else
  COT_PCT=0
fi

progress_str=""
[ "$IS_PATHOLOGY" = true ] && progress_str+="Path: $PAT_PROCESSED/$PAT_TOTAL (${PAT_PCT}%)  "
[ "$IS_CONSULT" = true ] && progress_str+="Cons: $CON_PROCESSED/$CON_TOTAL (${CON_PCT}%)  "
[ "$IS_TP" = true ] && progress_str+="OldTP: $TP_PROCESSED/$TP_TOTAL (${TP_PCT}%)  "
[ "$IS_PC" = true ] && progress_str+="PC: $PC_PROCESSED/$PC_TOTAL (${PC_PCT}%)  "
[ "$IS_COT" = true ] && progress_str+="CoT: $COT_PROCESSED/$COT_TOTAL (${COT_PCT}%)"

echo -e "\r[+] $progress_str  Done!"
echo "Logs are in $LOGFILE"
