#!/bin/bash
# Usage: ./run_prompt_experiment4.sh [report_type] [prompt_mode] [case_id]
#   report_type can be one of:
#     pathology_reports,
#     consultation_notes,
#     treatment_plan_outcomepred,
#     path_consult_reports,
#     cot_treatment_plan_outcomepred.
#
#   prompt_mode is optional (e.g., "combined" or "separated").
#   case_id is optional; if provided, exactly that file (matching the case ID) is processed;
#     otherwise, one random file is selected.
#
# Example:
#   bash run_prompt_experiment4.sh "path_consult_reports" "combined" "1219243"

#############################
# 1) Configuration
#############################
INPUT_DIR="/media/yujing/One Touch3/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt25"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
LOCAL_MODEL="llama3.3:latest"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/hnc_reports_agent4.py"

#############################
# 2) Trap for Process Termination
#############################
trap "echo 'Terminating all processes...'; kill -- -$$; exit 1" SIGINT SIGTERM

#############################
# 3) Parse Command-Line Arguments
#############################
if [ -z "$1" ]; then
  echo "Error: Please specify a report type."
  exit 1
else
  REPORT_TYPE="$1"
fi

if [ -z "$2" ]; then
  PROMPT_MODE=""
else
  PROMPT_MODE="$2"
fi

if [ -z "$3" ]; then
  CASE_ID=""
else
  CASE_ID="$3"
fi

echo "Selected report type: $REPORT_TYPE"
[ -n "$PROMPT_MODE" ] && echo "Using prompt mode: $PROMPT_MODE"
[ -n "$CASE_ID" ] && echo "Processing specified case ID: $CASE_ID" || echo "Processing one random case"

#############################
# 4) Run Python Script (Single-Case Experiment)
#############################
LOGFILE="/tmp/summarizer_prompt_experiment4.log"
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
  --single \
  ${CASE_ID:+--case_id "$CASE_ID"} \
  >"$LOGFILE" 2>&1 &

PID=$!

#############################
# 5) Spinner
#############################
spin='-\|/'
i=0
while kill -0 "$PID" 2>/dev/null; do
  i=$(( (i+1) % 4 ))
  spinChar=${spin:$i:1}
  echo -ne "\r[$spinChar] Running single-case experiment with PID=$PID ..."
  sleep 1
done

echo -e "\r[+] Done!"
echo "Logs are in $LOGFILE"
