#!/usr/bin/env python3
"""
HNC Summarizer with 4 Modes & Support for Ollama num_ctx

HNC (Head and Neck Cancer) Report Summarizer: 4 Experimental Modes + Large Context Support

This script processes head-and-neck cancer (HNC) patient reports by extracting a structured, 
30-line summary. The 30 fields include demographics, tumor staging, pathology findings, 
treatment history, and more. The script is capable of handling both:

1) **Combined** text (pathology and consultation reports already merged into a single file).
2) **Separate** text (one file under 'PathologyReports/', one under 'ConsultRedacted/', 
   automatically combined by the script).

It offers four "experiment modes," selectable by the --experiment_mode argument:

  **Mode 1**: Combined text + Single-step prompt
      - Reads from PathConsCombined/ 
      - Uses a single "combined" prompt to extract all 30 fields at once.

  **Mode 2**: Combined text + Two-step prompt
      - Reads from PathConsCombined/
      - Step 1: "extraction" prompt for partial fields
      - Step 2: "chain-of-thought (CoT)" prompt for the remaining fields
      - Merges the partial results for a complete 30-field summary

  **Mode 3**: Separate texts (Pathology + Consultation) + Single-step prompt
      - Looks in PathologyReports/ and ConsultRedacted/ for matching patient IDs
      - Summarizes each text individually using the single-step prompt
      - Merges final 30 fields so that if one text has "Not inferred" but the other has data, 
        the final output contains the inferred data

  **Mode 4**: Separate texts (Pathology + Consultation) + Two-step prompt
      - Same as Mode 3 but uses a two-step (extraction + CoT) approach on each text 
      - Merges the final results line-by-line

After summarizing, the script saves:

- **text_summaries/modeX/<patient_id>/summary.txt**:
  The final 30-line text summary (one line per field).
  
- **embeddings/modeX/<patient_id>/embedding.pkl**:
  A vector embedding (via Ollama, OpenAI, or Google embeddings) for the final summary text.

- **processing_times.csv** in the --output_dir:
  Captures inference time for each processed case (in milliseconds), the number of input 
  characters, and the approximate number of tokens.

Run the script directly, specifying:
  --prompts_dir: Where the JSON prompt files reside.
  --input_dir:   Parent directory containing PathologyReports/, ConsultRedacted/, PathConsCombined/.
  --output_dir:  Destination for the summarized text, embeddings, and processing times CSV.
  --experiment_mode: (1..4) which of the four modes to use.

Optionally, you can:
  --single:        Summarize only one random file/patient from each subfolder.
  --case_id:       Summarize a specific patient (filename w/o extension).
  --local_model:   Name of a local LLM (e.g. "llama3.3:latest") if using model_type=local.
  --ollama_context_size: If using Ollama, sets the num_ctx parameter (e.g. 128000 for large context).

Usage Example (Mode 2, combined text + two-step, large context):
  python hnc_reports_agent5.py \
    --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type local \
    --temperature 0.8 \
    --input_dir /media/yujing/OneTouch3/HNC_Reports \
    --output_dir /Data/Yujing/HNC_OutcomePred/Results \
    --embedding_model ollama \
    --local_model "llama3.3:latest" \
    --experiment_mode 2 \
    --ollama_context_size 128000

The final summary for each case is always a 30-line text with fields such as:
Sex, Anatomic_Site_of_Lesion, Pathological_TNM, ... ECOG_Performance_Status,
each in the form "FieldName: Value" or "Not inferred" if unknown.

We optionally set --ollama_context_size (num_ctx) for ChatOllama. Set for 128000 for llama3.3-70B.
"""

import os
import json
import re
import argparse
import logging
import pickle
import random
import time
import pandas as pd
from typing import List, Dict, Optional

# Transformers for token counting
from transformers import AutoTokenizer
try:
    # different llm models may have different tokenizers so we need to adjust if needed 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
except:
    raise ValueError("Could not load 'meta-llama/Llama-3.1-70B' tokenizer. Please adjust if needed.")

# LangChain
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

# Ollama + other possible LLM backends
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

###############################################################################
# 1. The 30 Fields
###############################################################################
ALL_FIELDS = [
    "Sex",
    "Anatomic_Site_of_Lesion",
    "Pathological_TNM",
    "Clinical_TNM",
    "Primary_Tumor_Size",
    "Tumor_Type_Differentiation",
    "Pathology_Details",
    "Lymph_Node_Status_Presence_Absence",
    "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes",
    "Lymph_Node_Status_Extranodal_Extension",
    "Resection_Margins",
    "p16_Status",
    "Immunohistochemical_profile",
    "EBER_Status",
    "Lymphovascular_Invasion_Status",
    "Perineural_Invasion_Status",
    "Smoking_History",
    "Alcohol_Consumption",
    "Pack_Years",
    "Patient_Symptoms_at_Presentation",
    "Recommendations",
    "Follow_Up_Plans",
    "HPV_Status",
    "Patient_History_Status_Prior_Conditions",
    "Patient_History_Status_Previous_Treatments",
    "Clinical_Assessments_Radiological_Lesions",
    "Clinical_Assessments_SUV_from_PET_scans",
    "Charlson_Comorbidity_Score",
    "Karnofsky_Performance_Status",
    "ECOG_Performance_Status"
]

###############################################################################
# 2. enforce_format(...) => ensures 30 lines
###############################################################################
def enforce_format(summary: str, fields: List[str]) -> str:
    lines = summary.splitlines()
    found = {}
    for fld in fields:
        pattern = rf"^{fld}:\s*"
        match_line = None
        for ln in lines:
            if re.match(pattern, ln.strip(), re.IGNORECASE):
                match_line = ln.strip()
                break
        if match_line:
            found[fld] = match_line
        else:
            found[fld] = f"{fld}: Not inferred"
    return "\n".join(found[f] for f in fields)

###############################################################################
# 3. Merge Summaries
###############################################################################
def merge_summaries(summary_a: str, summary_b: str, fields: List[str]) -> str:
    """
    If a field in summary_a is 'Not inferred' but summary_b has content, use summary_b, else summary_a.
    """
    def parse_lines(text: str) -> Dict[str, str]:
        d = {}
        for ln in text.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                d[k.strip()] = v.strip()
        return d

    da = parse_lines(summary_a)
    db = parse_lines(summary_b)
    merged = []
    for f in fields:
        va = da.get(f, "Not inferred")
        vb = db.get(f, "Not inferred")
        if va == "Not inferred" and vb != "Not inferred":
            merged.append(f"{f}: {vb}")
        else:
            merged.append(f"{f}: {va}")
    return "\n".join(merged)

###############################################################################
# 4. Summarizer Class
###############################################################################
class ReportSummarizer:
    def __init__(
        self,
        prompts_dir: str,
        model_type: str = "local",
        temperature: float = 0.8,
        embedding_model: str = "ollama",
        local_model: str = "llama3.3:latest",
        experiment_mode: int = 1,
        ollama_context_size: int = 4096
    ):
        """
        :param experiment_mode: 1..4 (the 4 modes).
        :param ollama_context_size: sets num_ctx for ChatOllama if model_type=local.
        """
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.embedding_model = embedding_model.lower()
        self.local_model = local_model
        self.experiment_mode = experiment_mode
        self.ollama_context_size = ollama_context_size

        if not os.path.isdir(prompts_dir):
            raise ValueError(f"prompts_dir={prompts_dir} is not valid.")
        self.prompts_dir = prompts_dir

        # Load prompts for single-step + two-step approach
        self.prompt_combined = self._load_prompt("prompt_path_consult_reports_combined.json")
        self.prompt_extraction = self._load_prompt("prompt_path_consult_reports_extraction.json")
        self.prompt_cot        = self._load_prompt("prompt_path_consult_reports_cot.json")

        # LLM
        if self.model_type == "local":
            # We pass num_ctx=... to ChatOllama to request large context
            self.model = ChatOllama(
                model=self.local_model,
                temperature=self.temperature,
                num_ctx=self.ollama_context_size  # <-- KEY (default is 2048, need to set this for LLMs even if the context window is much larger for that LLM itself like llama3.3-70 has has a context window of 128k)
            )
        elif self.model_type == "gpt":
            self.model = ChatOpenAI(model="gpt-4", temperature=self.temperature)
        elif self.model_type == "gemini":
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported model_type={self.model_type}")

        # Embeddings
        if self.embedding_model == "ollama":
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        elif self.embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif self.embedding_model == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        else:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            logger.warning(f"Unknown embedding_model {self.embedding_model}, defaulting to OllamaEmbeddings.")

        # Build runnables
        self.runnable_combined = self._make_llm_chain(self.prompt_combined) if self.prompt_combined else None
        self.runnable_extraction = self._make_llm_chain(self.prompt_extraction) if self.prompt_extraction else None
        self.runnable_cot = self._make_llm_chain(self.prompt_cot) if self.prompt_cot else None

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.prompts_dir, filename)
        if not os.path.isfile(path):
            logger.warning(f"Prompt file not found: {path}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lines = data.get("prompts", [])
        return "\n".join(lines)

    def _make_llm_chain(self, prompt_text: str) -> RunnableLambda:
        def chain_func(inputs: Dict[str, str]) -> Dict[str, str]:
            txt = inputs["context"]
            final_prompt = prompt_text.replace("{context}", txt)
            try:
                resp = self.model.invoke([HumanMessage(content=final_prompt)])
                return {"summary": resp.content.strip()}
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return {"summary": ""}
        return RunnableLambda(chain_func)

    # --- Summaries for Combined text (Modes 1 & 2) ---
    def summarize_combined_singlestep(self, text: str) -> str:
        if not self.runnable_combined:
            logger.warning("No combined prompt loaded.")
            return ""
        out = self.runnable_combined.invoke({"context": text})["summary"]
        return enforce_format(out, ALL_FIELDS)

    def summarize_combined_twostep(self, text: str) -> str:
        if not (self.runnable_extraction and self.runnable_cot):
            logger.warning("Missing extraction or CoT prompt => fallback to single-step.")
            return self.summarize_combined_singlestep(text)
        ex_summ = self.runnable_extraction.invoke({"context": text}).get("summary", "")
        cot_summ = self.runnable_cot.invoke({"context": text}).get("summary", "")
        merged = ex_summ + "\n" + cot_summ
        return enforce_format(merged, ALL_FIELDS)

    # --- Summaries for Separate text (Modes 3 & 4) ---
    def summarize_separate_singlestep(self, path_text: str, cons_text: str) -> str:
        sA = self.summarize_combined_singlestep(path_text)
        sB = self.summarize_combined_singlestep(cons_text)
        return merge_summaries(sA, sB, ALL_FIELDS)

    def summarize_separate_twostep(self, path_text: str, cons_text: str) -> str:
        sA = self.summarize_combined_twostep(path_text)
        sB = self.summarize_combined_twostep(cons_text)
        return merge_summaries(sA, sB, ALL_FIELDS)

    # --- Summarize one patient depending on experiment_mode ---
    def summarize_one_patient(self, path_text: str, cons_text: str) -> str:
        mode = self.experiment_mode
        if mode == 1:
            return self.summarize_combined_singlestep(path_text)
        elif mode == 2:
            return self.summarize_combined_twostep(path_text)
        elif mode == 3:
            return self.summarize_separate_singlestep(path_text, cons_text)
        elif mode == 4:
            return self.summarize_separate_twostep(path_text, cons_text)
        else:
            logger.warning(f"Invalid mode={mode}, returning empty.")
            return ""

    # --- Process entire dataset according to mode ---
    def process_reports(self, input_dir: str, output_dir: str, single: bool=False, case_id: Optional[str]=None):
        os.makedirs(output_dir, exist_ok=True)
        time_data = []

        if self.experiment_mode in [1, 2]:
            # Combined => read from PathConsCombined
            comb_dir = os.path.join(input_dir, "PathConsCombined")
            if not os.path.isdir(comb_dir):
                logger.error(f"Missing PathConsCombined: {comb_dir}")
                return
            all_files = []
            for root, _, fs in os.walk(comb_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        all_files.append(os.path.join(root, fname))
            if case_id:
                all_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] == case_id]
                if not all_files:
                    logger.warning(f"No file with case_id={case_id} in PathConsCombined.")
                    return
            if single and all_files:
                all_files = [random.choice(all_files)]

            for fp in all_files:
                pid = os.path.splitext(os.path.basename(fp))[0]
                logger.info(f"[Mode {self.experiment_mode}] Summarizing combined => {pid}")
                with open(fp, 'r', encoding='utf-8') as f:
                    text = f.read()

                num_chars = len(text)
                num_tokens = len(tokenizer.encode(text, add_special_tokens=False))

                st = time.time()
                summary = self.summarize_one_patient(text, "")
                et = time.time()
                elapsed_ms = int(round((et - st) * 1000))

                time_data.append({
                    "file": os.path.basename(fp),
                    "report_type": f"combined_mode{self.experiment_mode}",
                    "process_time_ms": elapsed_ms,
                    "num_input_characters": num_chars,
                    "num_input_tokens": num_tokens
                })

                if not summary:
                    logger.warning(f"No summary for {pid}.")
                    continue

                sub_text_dir = os.path.join(output_dir, "text_summaries", f"mode{self.experiment_mode}", pid)
                sub_emb_dir  = os.path.join(output_dir, "embeddings", f"mode{self.experiment_mode}", pid)
                os.makedirs(sub_text_dir, exist_ok=True)
                os.makedirs(sub_emb_dir, exist_ok=True)

                with open(os.path.join(sub_text_dir, "summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(summary)

                emb = self.embeddings.embed_documents([summary])[0]
                with open(os.path.join(sub_emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

        elif self.experiment_mode in [3, 4]:
            # Separate => read from PathologyReports + ConsultRedacted
            path_dir = os.path.join(input_dir, "PathologyReports")
            cons_dir = os.path.join(input_dir, "ConsultRedacted")

            if not os.path.isdir(path_dir) and not os.path.isdir(cons_dir):
                logger.error("Missing PathologyReports or ConsultRedacted.")
                return

            path_map = {}
            for root, _, fs in os.walk(path_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        pid = os.path.splitext(fname)[0]
                        path_map[pid] = os.path.join(root, fname)

            cons_map = {}
            for root, _, fs in os.walk(cons_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        pid = os.path.splitext(fname)[0]
                        cons_map[pid] = os.path.join(root, fname)

            all_pids = set(path_map.keys()) | set(cons_map.keys())
            if case_id:
                if case_id not in all_pids:
                    logger.warning(f"No matching case_id={case_id} in pathology or consult.")
                    return
                all_pids = {case_id}
            if single and len(all_pids) > 0:
                chosen = random.choice(list(all_pids))
                all_pids = {chosen}

            for pid in all_pids:
                path_text = ""
                cons_text = ""
                logger.info(f"[Mode {self.experiment_mode}] Summarizing separate => {pid}")

                if pid in path_map:
                    with open(path_map[pid], 'r', encoding='utf-8') as f:
                        path_text = f.read()

                if pid in cons_map:
                    with open(cons_map[pid], 'r', encoding='utf-8') as f:
                        cons_text = f.read()

                combined_input = path_text + "\n\n" + cons_text
                num_chars = len(combined_input)
                num_tokens = len(tokenizer.encode(combined_input, add_special_tokens=False))

                st = time.time()
                summary = self.summarize_one_patient(path_text, cons_text)
                et = time.time()
                elapsed_ms = int(round((et - st) * 1000))

                time_data.append({
                    "file": f"{pid}.txt",
                    "report_type": f"separate_mode{self.experiment_mode}",
                    "process_time_ms": elapsed_ms,
                    "num_input_characters": num_chars,
                    "num_input_tokens": num_tokens
                })

                if not summary:
                    logger.warning(f"No summary for pid={pid}.")
                    continue

                sub_text_dir = os.path.join(output_dir, "text_summaries", f"mode{self.experiment_mode}", pid)
                sub_emb_dir  = os.path.join(output_dir, "embeddings", f"mode{self.experiment_mode}", pid)
                os.makedirs(sub_text_dir, exist_ok=True)
                os.makedirs(sub_emb_dir, exist_ok=True)

                with open(os.path.join(sub_text_dir, "summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(summary)

                emb = self.embeddings.embed_documents([summary])[0]
                with open(os.path.join(sub_emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

        # Save time data
        if time_data:
            df = pd.DataFrame(time_data)
            csv_path = os.path.join(output_dir, "processing_times.csv")
            if os.path.isfile(csv_path):
                old_df = pd.read_csv(csv_path)
                merged = pd.concat([old_df, df], ignore_index=True)
                merged.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, index=False)
            logger.info(f"Saved timing info => {csv_path}")

###############################################################################
# 5. CLI
###############################################################################
def main():
    parser = argparse.ArgumentParser("HNC Summarizer with 4 modes + Ollama num_ctx")
    parser.add_argument("--prompts_dir", required=True, help="Folder containing JSON prompt files.")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend type.")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature.")
    parser.add_argument("--input_dir", required=True, help="Directory with PathologyReports, ConsultRedacted, PathConsCombined, etc.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--embedding_model", default="ollama", choices=["ollama","openai","google"], help="Embeddings model.")
    parser.add_argument("--local_model", default="llama3.3:latest", help="Which local model if --model_type=local.")
    parser.add_argument("--experiment_mode", type=int, default=1, help="1..4 => 4 experiment modes.")
    parser.add_argument("--single", action="store_true", help="If set, process only one random file/patient.")
    parser.add_argument("--case_id", default="", help="If provided, only that patient/file is processed.")
    parser.add_argument("--ollama_context_size", type=int, default=4096, help="num_ctx for Ollama large context window.")

    args = parser.parse_args()

    # Build Summarizer
    summ = ReportSummarizer(
        prompts_dir=args.prompts_dir,
        model_type=args.model_type,
        temperature=args.temperature,
        embedding_model=args.embedding_model,
        local_model=args.local_model,
        experiment_mode=args.experiment_mode,
        ollama_context_size=args.ollama_context_size
    )

    cid = args.case_id.strip() if args.case_id.strip() else None
    summ.process_reports(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        single=args.single,
        case_id=cid
    )

if __name__ == "__main__":
    # Let all child processes be in same group for easier kill
    os.setpgrp()
    main()

# usage example: mode 2, large context 
# python hnc_reports_agent5.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt29" \
#   --embedding_model ollama \
#   --local_model "llama3.3:latest" \
#   --experiment_mode 2 \
#   --ollama_context_size 128000 \
#   --case_id "1019973"

# python hnc_reports_agent5.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt29" \
#   --embedding_model ollama \
#   --local_model "llama3.3:latest" \
#   --experiment_mode 2 \
#   --ollama_context_size 4096 \
#   --case_id "1019973"


# 1130120 also has a long input text, with ollama_context_size 2048 I don't expect good results 

# python hnc_reports_agent5.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt30" \
#   --embedding_model ollama \
#   --local_model "llama3.3:latest" \
#   --experiment_mode 2 \
#   --ollama_context_size 2048 \
#   --case_id "1130120"

# since 1130120 has 2373 input tokens, including the prompt tokens, let's increase the ollama_context_size to 4096
# THIS WORKED WELL!!! 
# python hnc_reports_agent5.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt31" \
#   --embedding_model ollama \
#   --local_model "llama3.3:latest" \
#   --experiment_mode 2 \
#   --ollama_context_size 4096 \
#   --case_id "1130120"