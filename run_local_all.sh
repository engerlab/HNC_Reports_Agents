#!/bin/bash

# Bash script to run the summarizer with the local ollama models: parent input_reports: /Data/Yujing/HNC_OutcomePred/HNC_Reports

python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports3.py \
    --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type local \
    --temperature 0.8 \
    --input_dir /Data/Yujing/HNC_OutcomePred/HNC_Reports \
    --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp1 \
    --embedding_model ollama


# ollama llama3.3:latest was used here as the embedding model.

# cp -r /media/yujing/One\ Touch3/HNC_Reports /Data/Yujing/HNC_OutcomePred
