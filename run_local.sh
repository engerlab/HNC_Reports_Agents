#!/bin/bash

# Bash script to run the summarizer with the local Llama 3.2 model

python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports.py \
    --prompts_dir Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type local \
    --temperature 0.8 \
    --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
    --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/output_dir \
    --embedding_model ollama
