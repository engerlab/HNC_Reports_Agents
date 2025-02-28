#!/usr/bin/env python3
"""
HNC Summarizer with 4 Modes, Dynamic Context Window, Precompute Tokens, and Renamed Outputs

This script processes head-and-neck cancer (HNC) patient reports by extracting a
30-line structured summary (fields like Pathological_TNM, Clinical_TNM, etc.).
It supports four experiment modes:

  Mode 1 => Combined text + Single-step prompt
  Mode 2 => Combined text + Two-step prompt (extraction + CoT)
  Mode 3 => Separate texts (Pathology + Consultation) + Single-step prompt
  Mode 4 => Separate texts (Pathology + Consultation) + Two-step prompt

New Features:
  - Subfolder name is "mode1_combined_singlestep", "mode2_combined_twostep", etc.
  - The final summary file is saved as "path_consult_reports_summary.txt".
  - Appends all new runs to processing_times.csv if present.
  - A new "--precompute_tokens" flag to skip LLM calls and just record token counts in "precomputed_tokens.csv".
  - A new "--dynamic_ctx_window" option to auto-adjust num_ctx for Ollama on a per-case basis.
  - A new "--margin" parameter to add extra padding tokens to the dynamic context size.

Usage Examples:

1) Precompute tokens only (no summarization):
This scans:
    PathConsCombined/ (report_type = “combined”),
    PathologyReports/ (report_type = “pathology”),
    ConsultRedacted/ (report_type = “consult`),
and gathers token counts for each file (no LLM calls). It writes precomputed_tokens.csv in --output_dir.

python hnc_reports_agent6.py \
--input_dir "/media/yujing/OneTouch3/HNC_Reports" \
--output_dir "/Data/Yujing/HNC_OutcomePred/Results" \
--precompute_tokens

2) Mode 1: Combined + Single-Step Summaries
Processes .txt in PathConsCombined/, merges everything into a single step prompt. For all patients, no single-case:

python hnc_reports_agent6.py \
  --prompts_dir /path/to/prompts \
  --model_type local \
  --local_model "llama3.3:latest" \
  --embedding_model ollama \
  --input_dir "/path/to/HNC_Reports" \
  --output_dir "/path/to/results" \
  --experiment_mode 1 \
  --case_id "1130120" \
  --dynamic_ctx_window \
  --margin 500

Reads from PathConsCombined/1130120.txt (since Mode 1 = combined, single-step).
Dynamically calculates how many tokens the final single-step prompt might use.
If the total is less than --default_ctx (default 2048), it picks 2048. Otherwise, (needed_tokens + 500).
Appends a column like actual_context_size in processing_times.csv.

2) Mode 2: Combined + Two-Step Summaries: 
Processes .txt in PathConsCombined/ with a two-step approach (extraction + CoT). For a single case 1019973, using a large static context window of 64K tokens:

python hnc_reports_agent6.py \
  --prompts_dir /path/to/prompts \
  --model_type local \
  --local_model "llama3.3:latest" \
  --embedding_model ollama \
  --input_dir "/path/to/HNC_Reports" \
  --output_dir "/path/to/results" \
  --experiment_mode 2 \
  --case_id "1130120" \
  --dynamic_ctx_window \
  --margin 500

If no --ollama_context_size or --dynamic_ctx_window is provided, default is num_ctx=2048.
does two LLM calls (extraction + CoT). Each call will do an independent token count on “(prompt + text).”

4) Mode 3: Separate + Single-Step Summaries
Looks for PathologyReports/ + ConsultRedacted/. Summarizes each, merges fields. For all patients, but randomly picks one:

python hnc_reports_agent6.py \
  --prompts_dir /path/to/prompts \
  --model_type local \
  --local_model "llama3.3:latest" \
  --embedding_model ollama \
  --input_dir "/path/to/HNC_Reports" \
  --output_dir "/path/to/results" \
  --experiment_mode 3 \
  --case_id "1130120" \
  --dynamic_ctx_window \
  --margin 500

The script individually does single-step summarization for each text, merges the fields.
Dynamic context sizing is calculated for each text’s single-step prompt.

5) Mode 4: Separate + Two-Step Summaries
Similarly processes Pathology + Consultation, but two-step each. This time with dynamic context sizing:

python hnc_reports_agent6.py \
  --prompts_dir /path/to/prompts \
  --model_type local \
  --local_model "llama3.3:latest" \
  --embedding_model ollama \
  --input_dir "/path/to/HNC_Reports" \
  --output_dir "/path/to/results" \
  --experiment_mode 4 \
  --case_id "1130120" \
  --dynamic_ctx_window \
  --margin 500


For each two-step call, the script calculates the number of tokens. If it’s below default_ctx=2048, it uses 2048. Otherwise, it uses (prompt_tokens + margin).
mode4_separate_twostep/... subfolders.

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

# 1) Tokenizer
from transformers import AutoTokenizer
try:
    # For demonstration, using a hypothetical LLaMA-3.1-70B or similar
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
except:
    raise ValueError("Could not load 'meta-llama/Llama-3.1-70B' tokenizer. Adjust if needed.")

# 2) LangChain
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

# 3) LLM Backends
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
# 2. Utility to Enforce 30-Field Format
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
    If summary_a says 'Not inferred' for a field but summary_b has data, use summary_b. Else summary_a.
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
# 4. Summarizer
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
        ollama_context_size: int = 2048,
        dynamic_ctx_window: bool = False,
        default_ctx: int = 2048,
        margin: int = 500
    ):
        """
        :param experiment_mode: int in {1..4}
        :param ollama_context_size: default num_ctx if not using dynamic
        :param dynamic_ctx_window: if True, dynamically compute num_ctx per case
        :param default_ctx: fallback if sum tokens < default_ctx
        :param margin: extra tokens to add if we surpass default_ctx
        """
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.embedding_model = embedding_model.lower()
        self.local_model = local_model
        self.experiment_mode = experiment_mode
        self.dynamic_ctx_window = dynamic_ctx_window
        self.ollama_context_size = ollama_context_size
        self.default_ctx = default_ctx
        self.margin = margin

        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Invalid prompts_dir: {prompts_dir}")
        self.prompts_dir = prompts_dir

        # Prompts
        self.prompt_combined = self._load_prompt("prompt_path_consult_reports_combined.json")
        self.prompt_extraction = self._load_prompt("prompt_path_consult_reports_extraction.json")
        self.prompt_cot        = self._load_prompt("prompt_path_consult_reports_cot.json")

        # We'll set self.model once we know the context window for each call if dynamic. 
        # If not dynamic, we create it right now with ollama_context_size. 
        if self.model_type == "local" and (not self.dynamic_ctx_window):
            self.model = ChatOllama(
                model=self.local_model,
                temperature=self.temperature,
                num_ctx=self.ollama_context_size
            )
        elif self.model_type == "local" and self.dynamic_ctx_window:
            # We'll create a new ChatOllama instance for each call, or re-init 
            # after we measure tokens. We'll do that in `_make_llm_chain`.
            self.model = None
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
            logger.warning(f"Unknown embedding_model {self.embedding_model}, defaulting to OllamaEmbeddings.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build runnables
        self.runnable_combined = self._make_llm_chain(self.prompt_combined) if self.prompt_combined else None
        self.runnable_extraction = self._make_llm_chain(self.prompt_extraction) if self.prompt_extraction else None
        self.runnable_cot = self._make_llm_chain(self.prompt_cot) if self.prompt_cot else None

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.prompts_dir, filename)
        if not os.path.isfile(path):
            logger.warning(f"No prompt file: {path}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "\n".join(data.get("prompts", []))

    def _estimate_prompt_tokens(self, prompt_text: str, input_text: str) -> int:
        """
        Roughly estimate how many tokens the final prompt will have.
        """
        final_prompt = prompt_text.replace("{context}", input_text)
        return len(tokenizer.encode(final_prompt, add_special_tokens=False))

    def _make_llm_chain(self, prompt_text: str) -> RunnableLambda:
        """
        Build a function that, given {"context": some_text}, calls the LLM with (prompt + text).
        If dynamic_ctx_window is set, we compute the needed tokens and re-init ChatOllama as needed.
        """
        def chain_func(inputs: Dict[str, str]) -> Dict[str, str]:
            txt = inputs["context"]

            # 1) Construct final prompt
            final_prompt = prompt_text.replace("{context}", txt)
            # 2) If dynamic, estimate tokens. Then re-init model with new num_ctx
            if self.model_type == "local" and self.dynamic_ctx_window:
                needed_tokens = len(tokenizer.encode(final_prompt, add_special_tokens=False))
                # If we are in two-step mode (ex. experiment_mode=2 or 4), we might do 2 calls.
                # This chain_func is just one step. We'll do the sum in the separate calls. 
                # For simplicity, we do each step individually. 
                if needed_tokens < self.default_ctx:
                    ctx_size = self.default_ctx
                else:
                    ctx_size = needed_tokens + self.margin

                logger.debug(f"[Dynamic ctx] needed={needed_tokens}, using ctx={ctx_size}")
                self.model = ChatOllama(
                    model=self.local_model,
                    temperature=self.temperature,
                    num_ctx=ctx_size
                )

            # 3) Actually call the model
            try:
                resp = self.model.invoke([HumanMessage(content=final_prompt)])
                return {"summary": resp.content.strip()}
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return {"summary": ""}
        return RunnableLambda(chain_func)

    # --- Summaries for Combined text ---
    def summarize_combined_singlestep(self, text: str) -> str:
        if not self.runnable_combined:
            logger.warning("No combined prompt loaded!")
            return ""
        out = self.runnable_combined.invoke({"context": text})["summary"]
        return enforce_format(out, ALL_FIELDS)

    def summarize_combined_twostep(self, text: str) -> str:
        if not (self.runnable_extraction and self.runnable_cot):
            logger.warning("Missing extraction or CoT prompt => fallback single-step.")
            return self.summarize_combined_singlestep(text)
        ex_summ = self.runnable_extraction.invoke({"context": text}).get("summary", "")
        cot_summ = self.runnable_cot.invoke({"context": text}).get("summary", "")
        merged = ex_summ + "\n" + cot_summ
        return enforce_format(merged, ALL_FIELDS)

    # --- Summaries for Separate text ---
    def summarize_separate_singlestep(self, path_text: str, cons_text: str) -> str:
        sum_path = self.summarize_combined_singlestep(path_text)
        sum_cons = self.summarize_combined_singlestep(cons_text)
        return merge_summaries(sum_path, sum_cons, ALL_FIELDS)

    def summarize_separate_twostep(self, path_text: str, cons_text: str) -> str:
        sum_path = self.summarize_combined_twostep(path_text)
        sum_cons = self.summarize_combined_twostep(cons_text)
        return merge_summaries(sum_path, sum_cons, ALL_FIELDS)

    def summarize_one_patient(self, path_text: str, cons_text: str) -> str:
        if self.experiment_mode == 1:
            return self.summarize_combined_singlestep(path_text)
        elif self.experiment_mode == 2:
            return self.summarize_combined_twostep(path_text)
        elif self.experiment_mode == 3:
            return self.summarize_separate_singlestep(path_text, cons_text)
        elif self.experiment_mode == 4:
            return self.summarize_separate_twostep(path_text, cons_text)
        else:
            logger.warning(f"Unknown experiment_mode={self.experiment_mode}")
            return ""

    ###########################################################################
    #  (A) Precompute tokens if requested
    ###########################################################################
    def precompute_tokens(self, input_dir: str, output_dir: str):
        """
        Gathers all .txt files from:
          - PathConsCombined/
          - PathologyReports/
          - ConsultRedacted/
        For each file, store:
          (report_type, patient_id, num_chars, num_tokens)
        in 'precomputed_tokens.csv'.
        """
        records = []

        # 1) PathConsCombined => "combined" 
        combined_dir = os.path.join(input_dir, "PathConsCombined")
        if os.path.isdir(combined_dir):
            for root, _, fs in os.walk(combined_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        pid = os.path.splitext(fname)[0]
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        n_chars = len(txt)
                        n_tokens = len(tokenizer.encode(txt, add_special_tokens=False))
                        records.append({
                            "report_type": "combined",
                            "patient_id": pid,
                            "file": fname,
                            "num_chars": n_chars,
                            "num_tokens": n_tokens
                        })

        # 2) PathologyReports => "pathology"
        path_dir = os.path.join(input_dir, "PathologyReports")
        if os.path.isdir(path_dir):
            for root, _, fs in os.walk(path_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        pid = os.path.splitext(fname)[0]
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        n_chars = len(txt)
                        n_tokens = len(tokenizer.encode(txt, add_special_tokens=False))
                        records.append({
                            "report_type": "pathology",
                            "patient_id": pid,
                            "file": fname,
                            "num_chars": n_chars,
                            "num_tokens": n_tokens
                        })

        # 3) ConsultRedacted => "consult"
        cons_dir = os.path.join(input_dir, "ConsultRedacted")
        if os.path.isdir(cons_dir):
            for root, _, fs in os.walk(cons_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        pid = os.path.splitext(fname)[0]
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        n_chars = len(txt)
                        n_tokens = len(tokenizer.encode(txt, add_special_tokens=False))
                        records.append({
                            "report_type": "consult",
                            "patient_id": pid,
                            "file": fname,
                            "num_chars": n_chars,
                            "num_tokens": n_tokens
                        })

        if not records:
            logger.warning("No .txt files found for precompute.")
            return

        df = pd.DataFrame(records)
        outpath = os.path.join(output_dir, "precomputed_tokens.csv")
        df.to_csv(outpath, index=False)
        logger.info(f"Precomputed tokens saved => {outpath}")

    ###########################################################################
    #  (B) Normal Summarization
    ###########################################################################
    def process_reports(
        self,
        input_dir: str,
        output_dir: str,
        single: bool=False,
        case_id: Optional[str]=None
    ):
        os.makedirs(output_dir, exist_ok=True)
        time_data = []

        # Determine subfolder name based on experiment_mode
        if self.experiment_mode == 1:
            subfolder = "mode1_combined_singlestep"
            combined_needed = True
        elif self.experiment_mode == 2:
            subfolder = "mode2_combined_twostep"
            combined_needed = True
        elif self.experiment_mode == 3:
            subfolder = "mode3_separate_singlestep"
            combined_needed = False
        elif self.experiment_mode == 4:
            subfolder = "mode4_separate_twostep"
            combined_needed = False
        else:
            logger.error(f"Invalid experiment_mode={self.experiment_mode}")
            return

        # (1) Combined text => PathConsCombined
        if combined_needed:
            comb_dir = os.path.join(input_dir, "PathConsCombined")
            if not os.path.isdir(comb_dir):
                logger.error(f"Missing PathConsCombined => {comb_dir}")
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

                # input stats
                num_chars = len(text)
                num_tokens = len(tokenizer.encode(text, add_special_tokens=False))

                st = time.time()
                summary = self.summarize_one_patient(text, "")
                et = time.time()
                elapsed_ms = int(round((et - st) * 1000))

                time_data.append({
                    "file": os.path.basename(fp),
                    "report_type": f"{subfolder}",
                    "process_time_ms": elapsed_ms,
                    "num_input_characters": num_chars,
                    "num_input_tokens": num_tokens,
                    "patient_id": pid
                })

                if not summary:
                    logger.warning(f"No summary for {pid}")
                    continue

                sub_text_dir = os.path.join(output_dir, "text_summaries", subfolder, pid)
                sub_emb_dir  = os.path.join(output_dir, "embeddings", subfolder, pid)
                os.makedirs(sub_text_dir, exist_ok=True)
                os.makedirs(sub_emb_dir, exist_ok=True)

                # Save final summary
                with open(os.path.join(sub_text_dir, "path_consult_reports_summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(summary)

                # Save embedding
                emb = self.embeddings.embed_documents([summary])[0]
                with open(os.path.join(sub_emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

        # (2) Separate => PathologyReports + ConsultRedacted
        else:
            path_dir = os.path.join(input_dir, "PathologyReports")
            cons_dir = os.path.join(input_dir, "ConsultRedacted")
            if not (os.path.isdir(path_dir) or os.path.isdir(cons_dir)):
                logger.error("Missing PathologyReports or ConsultRedacted.")
                return

            path_map = {}
            if os.path.isdir(path_dir):
                for root, _, fs in os.walk(path_dir):
                    for fname in fs:
                        if fname.endswith(".txt"):
                            pid = os.path.splitext(fname)[0]
                            path_map[pid] = os.path.join(root, fname)

            cons_map = {}
            if os.path.isdir(cons_dir):
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
                logger.info(f"[Mode {self.experiment_mode}] Summarizing separate => {pid}")
                path_text = ""
                cons_text = ""
                if pid in path_map:
                    with open(path_map[pid], 'r', encoding='utf-8') as f:
                        path_text = f.read()
                if pid in cons_map:
                    with open(cons_map[pid], 'r', encoding='utf-8') as f:
                        cons_text = f.read()

                combined = path_text + "\n\n" + cons_text
                num_chars = len(combined)
                num_tokens = len(tokenizer.encode(combined, add_special_tokens=False))

                st = time.time()
                summary = self.summarize_one_patient(path_text, cons_text)
                et = time.time()
                elapsed_ms = int(round((et - st) * 1000))

                time_data.append({
                    "file": f"{pid}.txt",
                    "report_type": f"{subfolder}",
                    "process_time_ms": elapsed_ms,
                    "num_input_characters": num_chars,
                    "num_input_tokens": num_tokens,
                    "patient_id": pid
                })

                if not summary:
                    logger.warning(f"No summary for pid={pid}")
                    continue

                sub_text_dir = os.path.join(output_dir, "text_summaries", subfolder, pid)
                sub_emb_dir = os.path.join(output_dir, "embeddings", subfolder, pid)
                os.makedirs(sub_text_dir, exist_ok=True)
                os.makedirs(sub_emb_dir, exist_ok=True)

                with open(os.path.join(sub_text_dir, "path_consult_reports_summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(summary)

                emb = self.embeddings.embed_documents([summary])[0]
                with open(os.path.join(sub_emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

        # Finally save the timing data
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
    parser = argparse.ArgumentParser("HNC Summarizer with 4 modes, dynamic ctx, and precompute tokens.")
    parser.add_argument("--prompts_dir", required=True, help="Directory with prompt JSONs.")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--input_dir", required=True, help="Parent folder with PathConsCombined, PathologyReports, etc.")
    parser.add_argument("--output_dir", required=True, help="Where to store text_summaries, embeddings, CSVs.")
    parser.add_argument("--embedding_model", default="ollama", choices=["ollama","openai","google"], help="Emb model.")
    parser.add_argument("--local_model", default="llama3.3:latest", help="Local model name if model_type=local.")
    parser.add_argument("--experiment_mode", type=int, default=1, help="1..4 => the four modes.")
    parser.add_argument("--single", action="store_true", help="If set, pick one random file/patient per folder.")
    parser.add_argument("--case_id", default="", help="If provided, only that patient/file is processed.")
    parser.add_argument("--ollama_context_size", type=int, default=2048, 
                        help="Base num_ctx for ChatOllama (if dynamic_ctx_window is off).")
    parser.add_argument("--dynamic_ctx_window", action="store_true",
                        help="If set, we auto-calculate tokens for each LLM call and set num_ctx accordingly.")
    parser.add_argument("--default_ctx", type=int, default=2048,
                        help="If dynamic_ctx_window, use this if needed tokens < default_ctx.")
    parser.add_argument("--margin", type=int, default=500,
                        help="Extra tokens to add if needed tokens exceed default_ctx.")
    parser.add_argument("--precompute_tokens", action="store_true",
                        help="If set, skip summarization. Just gather token counts for all text files, store in CSV.")

    args = parser.parse_args()

    # Create Summarizer
    summarizer = ReportSummarizer(
        prompts_dir=args.prompts_dir,
        model_type=args.model_type,
        temperature=args.temperature,
        embedding_model=args.embedding_model,
        local_model=args.local_model,
        experiment_mode=args.experiment_mode,
        ollama_context_size=args.ollama_context_size,
        dynamic_ctx_window=args.dynamic_ctx_window,
        default_ctx=args.default_ctx,
        margin=args.margin
    )

    # If precompute_tokens => skip normal summarization
    if args.precompute_tokens:
        os.makedirs(args.output_dir, exist_ok=True)
        summarizer.precompute_tokens(args.input_dir, args.output_dir)
        return

    cid = args.case_id.strip() if args.case_id.strip() else None

    # Normal summarization
    summarizer.process_reports(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        single=args.single,
        case_id=cid
    )

if __name__ == "__main__":
    os.setpgrp()
    main()


# Usage examples:

# 1) Precompute tokens only (no summarization):

# python hnc_reports_agent6.py \
# --input_dir "/media/yujing/OneTouch3/HNC_Reports" \
# --output_dir "/Data/Yujing/HNC_OutcomePred/Results" \
# --precompute_tokens

# Mode1: Combined + Single-Step Summaries

# python hnc_reports_agent6.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --local_model "llama3.3:latest" \
#   --embedding_model ollama \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt33" \
#   --experiment_mode 1 \
#   --case_id "1130120" \
#   --dynamic_ctx_window \
#   --margin 500

# Mode 2: Combined + Two-Step Summaries: 

# python hnc_reports_agent6.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --local_model "llama3.3:latest" \
#   --embedding_model ollama \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt34" \
#   --experiment_mode 2 \
#   --case_id "1130120" \
#   --dynamic_ctx_window \
#   --margin 500

# Mode 3: Separate + Single-Step Summaries

# python hnc_reports_agent6.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --local_model "llama3.3:latest" \
#   --embedding_model ollama \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt35" \
#   --experiment_mode 3 \
#   --case_id "1130120" \
#   --dynamic_ctx_window \
#   --margin 500

# Mode 4: Separate + Two-Step Summaries

# python hnc_reports_agent6.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --local_model "llama3.3:latest" \
#   --embedding_model ollama \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt36" \
#   --experiment_mode 4 \
#   --case_id "1130120" \
#   --dynamic_ctx_window \
#   --margin 500