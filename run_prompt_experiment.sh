#!/bin/bash
# Usage: ./run_prompt_experiment.sh [report_type] [case_id] [prompt_mode]
# report_type can be a comma-separated list of:
#   pathology_reports,
#   consultation_notes,
#   treatment_plan_outcomepred,
#   path_consult_reports,
#   cot_treatment_plan_outcomepred,
#   or "all" (default).
#
# case_id is optional (filename without extension) to process a specific case.
# prompt_mode is optional (e.g., "combined" or leave empty for default).
#
# Examples:
#   bash run_prompt_experiment.sh "path_consult_reports" "1145281" "combined"
#   bash run_prompt_experiment.sh "path_consult_reports" "" "combined"  # processes one random file per subfolder

#############################
# 1) Configuration
#############################
INPUT_DIR="/media/yujing/One Touch3/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPrompt"
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
LOCAL_MODEL="llama3.3:latest"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/hnc_reports_agent3.py"

#############################
# 2) Parse Command-Line Arguments
#############################
# First argument: report_type (default "all")
if [ -z "$1" ] || [ "$1" == "all" ]; then
  REPORT_TYPE="all"
else
  REPORT_TYPE="$1"
fi

# Second argument: case_id (optional)
CASE_ID=""
if [ ! -z "$2" ]; then
  CASE_ID="$2"
fi

# Third argument: prompt_mode (optional)
if [ -z "$3" ]; then
  PROMPT_MODE=""
else
  PROMPT_MODE="$3"
fi

# Convert report_type to lowercase
REPORT_TYPE="${REPORT_TYPE,,}"

echo "Starting prompt experiment with report type: $REPORT_TYPE"
if [ -n "$CASE_ID" ]; then
  echo "Processing specified case ID: $CASE_ID"
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
    --case_id "$CASE_ID"
else
  echo "Processing one random file per subfolder (single case experiment)"
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
    --single
fi
