from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B") # or whatever model name you used with Ollama

with open("/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts/prompt_path_consult_reports_extraction.json", "r") as f: # or whatever prompt file
    prompt_data = json.load(f)
    prompt_text = prompt_data["prompt"] # Or however the prompt is stored in the JSON

prompt_tokens = tokenizer(prompt_text).input_ids
prompt_token_count = len(prompt_tokens)

print(f"Prompt token count: {prompt_token_count}")
