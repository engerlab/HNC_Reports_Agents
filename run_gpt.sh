#!/bin/bash

# Bash script to run the summarizer with the OpenAI GPT model

python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports.py \
    --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type gpt \
    --temperature 0.7 \
    --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
    --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/output_dir \
    --embedding_model openai
