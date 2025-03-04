#!/bin/bash
# Usage: ./run_local_all8.sh [experiment_mode] [ollama_context_size] [only_above]
#
#   experiment_mode     : 1..4 (which of the 4 modes to run)
#   ollama_context_size : integer, e.g. 4096 or 5000 (default 2048 if omitted)
#   only_above          : integer threshold for subset filtering (default=0 means no filtering)
#
# Example:
#   ./run_local_all8.sh 2 5000 1500
#     => runs experiment_mode=2, sets a fixed context of 5000 tokens,
#        filters only rows where extr_total_needed>1500 or cot_total_needed>1500
#        from the default subset CSV set below.
#
##############################################################################
# 1) Configuration
##############################################################################
PROMPTS_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts"
MODEL_TYPE="local"
TEMPERATURE="0.8"
EMBEDDING_MODEL="ollama"
LOCAL_MODEL="llama3.3:latest"
PYTHON_SCRIPT="/Data/Yujing/HNC_OutcomePred/Reports_Agents/hnc_reports_agent8.py"

INPUT_DIR="/media/yujing/One Touch3/HNC_Reports"
OUTPUT_DIR="/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt38"

# By default, we set this to the precompute_tokens_mode2.csv path. If you want to override,
# you can either edit here or set SUBSET_CSV externally in your environment before calling the script.
: "${SUBSET_CSV:="/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode2.csv"}"

##############################################################################
# 2) Trap for Process Termination
##############################################################################
trap "echo 'Terminating all processes...'; kill -- -$$; exit 1" SIGINT SIGTERM

##############################################################################
# 3) Parse Command-Line
##############################################################################
EXPMODE="$1"
OLLA_CTX="$2"
ONLY_ABOVE="$3"

# Fallback defaults if not provided
[ -z "$EXPMODE" ] && EXPMODE="1"
[ -z "$OLLA_CTX" ] && OLLA_CTX="2048"
[ -z "$ONLY_ABOVE" ] && ONLY_ABOVE="0"

echo "Experiment mode: $EXPMODE"
echo "ollama_context_size: $OLLA_CTX"
echo "only_above threshold: $ONLY_ABOVE"
echo "subset_precompute_csv: $SUBSET_CSV"

##############################################################################
# 4) Run Python Script
##############################################################################
LOGFILE="/tmp/summarizer_progress8.log"
rm -f "$LOGFILE"

python "$PYTHON_SCRIPT" \
  --prompts_dir "$PROMPTS_DIR" \
  --model_type "$MODEL_TYPE" \
  --temperature "$TEMPERATURE" \
  --local_model "$LOCAL_MODEL" \
  --embedding_model "$EMBEDDING_MODEL" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --experiment_mode "$EXPMODE" \
  --ollama_context_size "$OLLA_CTX" \
  --subset_precompute_csv "$SUBSET_CSV" \
  --only_above "$ONLY_ABOVE" \
  >"$LOGFILE" 2>&1 &

PID=$!

##############################################################################
# 5) Spinner
##############################################################################
spin='-\|/'
i=0
while kill -0 "$PID" 2>/dev/null; do
  i=$(( (i+1) % 4 ))
  spinChar=${spin:$i:1}
  echo -ne "\r[$spinChar] Experiment mode $EXPMODE is running (PID=$PID)..."
  sleep 1
done

echo -e "\r[+] Done!"
echo "Logs are in $LOGFILE"
