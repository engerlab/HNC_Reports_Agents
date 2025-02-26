#!/bin/bash
# Usage: ./run_local_all5.sh [report_type] [prompt_mode]
#   report_type: Comma-separated list of report types to process. Options:
#       pathology_reports, consultation_notes, treatment_plan_outcomepred, path_consult_reports, cot_treatment_plan_outcomepred, or "all" (default).
#   prompt_mode: Optional prompt suffix (e.g., "combined" or "separated"). Leave empty for default.
#
# Example:
#   bash run_local_all5.sh "path_consult_reports" "combined"

#############################
# 1) Configuration
#############################
INPUT_DIR="/media/yujing/One Touch3/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp13"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
LOCAL_MODEL="llama3.3:latest"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/hnc_reports_agent5.py"

#############################
# 2) Trap for Process Termination
#############################
trap "echo 'Terminating all processes...'; kill -- -$$; exit 1" SIGINT SIGTERM

#############################
# 3) Parse Command-Line Arguments
#############################
if [ -z "$1" ] || [ "$1" == "all" ]; then
  REPORT_TYPE="all"
else
  REPORT_TYPE="$1"
fi

if [ -z "$2" ]; then
  PROMPT_MODE=""
else
  PROMPT_MODE="$2"
fi

echo "Selected report types: $REPORT_TYPE"
[ -n "$PROMPT_MODE" ] && echo "Using prompt mode: $PROMPT_MODE"

#############################
# 4) Run Python Script (Full Processing)
#############################
LOGFILE="/tmp/summarizer_progress5.log"
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
  --prompt_mode "$PROMPT_MODE" \
  >"$LOGFILE" 2>&1 &

PID=$!

#############################
# 5) Spinner and Dynamic Progress
#############################
spin='-\|/'
i=0
while kill -0 "$PID" 2>/dev/null; do
  i=$(( (i+1) % 4 ))
  spinChar=${spin:$i:1}
  progress_str=""
  if [[ "$REPORT_TYPE" =~ "pathology_reports" || "$REPORT_TYPE" == "all" ]]; then
    PAT_PROCESSED=$(grep -c "Processing file: .* for report type: pathology_reports" "$LOGFILE")
    PAT_TOTAL=$(find "$INPUT_DIR/PathologyReports" -type f -name '*.txt' 2>/dev/null | wc -l)
    PAT_PCT=$(( PAT_TOTAL > 0 ? 100 * PAT_PROCESSED / PAT_TOTAL : 0 ))
    progress_str+="Path: $PAT_PROCESSED/$PAT_TOTAL (${PAT_PCT}%)  "
  fi
  if [[ "$REPORT_TYPE" =~ "consultation_notes" || "$REPORT_TYPE" == "all" ]]; then
    CON_PROCESSED=$(grep -c "Processing file: .* for report type: consultation_notes" "$LOGFILE")
    CON_TOTAL=$(find "$INPUT_DIR/ConsultRedacted" -type f -name '*.txt' 2>/dev/null | wc -l)
    CON_PCT=$(( CON_TOTAL > 0 ? 100 * CON_PROCESSED / CON_TOTAL : 0 ))
    progress_str+="Cons: $CON_PROCESSED/$CON_TOTAL (${CON_PCT}%)  "
  fi
  if [[ "$REPORT_TYPE" =~ "treatment_plan_outcomepred" || "$REPORT_TYPE" == "all" ]]; then
    TP_PROCESSED=$(grep -c "Processing file: .* for report type: treatment_plan_outcomepred" "$LOGFILE")
    TP_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
    TP_PCT=$(( TP_TOTAL > 0 ? 100 * TP_PROCESSED / TP_TOTAL : 0 ))
    progress_str+="TP: $TP_PROCESSED/$TP_TOTAL (${TP_PCT}%)  "
  fi
  if [[ "$REPORT_TYPE" =~ "path_consult_reports" || "$REPORT_TYPE" == "all" ]]; then
    PC_PROCESSED=$(grep -c "Processing file: .* for report type: path_consult_reports" "$LOGFILE")
    PC_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
    PC_PCT=$(( PC_TOTAL > 0 ? 100 * PC_PROCESSED / PC_TOTAL : 0 ))
    progress_str+="PC: $PC_PROCESSED/$PC_TOTAL (${PC_PCT}%)  "
  fi
  if [[ "$REPORT_TYPE" =~ "cot_treatment_plan_outcomepred" || "$REPORT_TYPE" == "all" ]]; then
    COT_PROCESSED=$(grep -c "Processing file: .* for report type: cot_treatment_plan_outcomepred" "$LOGFILE")
    COT_TOTAL=$(find "$INPUT_DIR/PathConsCombined" -type f -name '*.txt' 2>/dev/null | wc -l)
    COT_PCT=$(( COT_TOTAL > 0 ? 100 * COT_PROCESSED / COT_TOTAL : 0 ))
    progress_str+="CoT: $COT_PROCESSED/$COT_TOTAL (${COT_PCT}%)"
  fi
  echo -ne "\r[$spinChar] $progress_str"
  sleep 1
done

echo -e "\r[+] $progress_str  Done!"
echo "Logs are in $LOGFILE"
