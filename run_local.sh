#!/bin/bash

# Bash script to run the summarizer with the local ollama models

python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports3.py \
    --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type local \
    --temperature 0.8 \
    --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
    --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Experiments/Exp5 \
    --embedding_model ollama

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports3.py \
#     --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#     --model_type local \
#     --temperature 0.8 \
#     --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
#     --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Experiments/Exp4 \
#     --embedding_model ollama

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports3.py \
#     --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#     --model_type local \
#     --temperature 0.8 \
#     --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
#     --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Experiments/Exp3 \
#     --embedding_model ollama

# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports2.py \
#     --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#     --model_type local \
#     --temperature 0.8 \
#     --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
#     --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/Experiments/Exp2 \
#     --embedding_model ollama

# ollama llama3.3:latest was used here as the embedding model.


# python /Data/Yujing/HNC_OutcomePred/Reports_Agents/summarize_reports.py \
#     --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#     --model_type local \
#     --temperature 0.8 \
#     --input_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/input_reports \
#     --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/output_dir \
#     --embedding_model ollama

# ollama llama3.3:latest was used here as the embedding model.