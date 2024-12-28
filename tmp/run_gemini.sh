#!/bin/bash

# Bash script to run the summarizer with the Google Gemini model

python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports.py \
    --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type gemini \
    --temperature 0.7 \
    --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
    --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/output_dir \
    --embedding_model google
