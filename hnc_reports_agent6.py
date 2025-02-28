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

from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
except:
    raise ValueError("Could not load 'meta-llama/Llama-3.1-70B' tokenizer. Please adjust if needed.")

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

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
# 2. Format Enforcement
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
# 3. Merge Summaries for Separate-Text
###############################################################################
def merge_summaries(summary_a: str, summary_b: str, fields: List[str]) -> str:
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
        dynamic_ctx_window: bool = False,
        default_ctx: int = 2048,
        margin: int = 500,
        ollama_context_size: int = 2048
    ):
        """
        experiment_mode: 1..4
        dynamic_ctx_window: if True, we recalc num_ctx per call
        default_ctx: minimum or fallback context size
        margin: add this many tokens if needed tokens exceed default_ctx
        ollama_context_size: static fallback if not dynamic
        """
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.embedding_model = embedding_model.lower()
        self.local_model = local_model
        self.experiment_mode = experiment_mode
        self.dynamic_ctx_window = dynamic_ctx_window
        self.default_ctx = default_ctx
        self.margin = margin
        self.ollama_context_size = ollama_context_size

        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Invalid prompts_dir: {prompts_dir}")
        self.prompts_dir = prompts_dir

        # Load relevant prompts
        self.prompt_combined = self._load_prompt("prompt_path_consult_reports_combined.json")
        self.prompt_extraction = self._load_prompt("prompt_path_consult_reports_extraction.json")
        self.prompt_cot        = self._load_prompt("prompt_path_consult_reports_cot.json")

        # LLM
        # If not dynamic, create model now. Otherwise, we do it ad-hoc inside each call.
        if self.model_type == "local" and (not self.dynamic_ctx_window):
            self.model = ChatOllama(
                model=self.local_model,
                temperature=self.temperature,
                num_ctx=self.ollama_context_size
            )
        elif self.model_type == "local" and self.dynamic_ctx_window:
            self.model = None  # We'll init dynamically
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
            logging.warning(f"Unknown embedding_model={self.embedding_model}, defaulting to Ollama.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build runnables
        self.runnable_combined = self._make_llm_chain(self.prompt_combined) if self.prompt_combined else None
        self.runnable_extraction = self._make_llm_chain(self.prompt_extraction) if self.prompt_extraction else None
        self.runnable_cot = self._make_llm_chain(self.prompt_cot) if self.prompt_cot else None

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.prompts_dir, filename)
        if not os.path.isfile(path):
            logger.warning(f"No prompt found for {filename}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "\n".join(data.get("prompts", []))

    def _estimate_prompt_tokens(self, prompt_text: str, text: str) -> int:
        final = prompt_text.replace("{context}", text)
        return len(tokenizer.encode(final, add_special_tokens=False))

    def _init_model_for_tokens(self, needed_tokens: int) -> int:
        """
        Return final ctx_size used. Then re-init self.model if dynamic.
        """
        if not self.dynamic_ctx_window:
            return self.ollama_context_size
        # dynamic approach
        if needed_tokens < self.default_ctx:
            ctx_size = self.default_ctx
        else:
            ctx_size = needed_tokens + self.margin
        logger.debug(f"[Dynamic ctx] needed={needed_tokens}, used={ctx_size}")
        self.model = ChatOllama(
            model=self.local_model,
            temperature=self.temperature,
            num_ctx=ctx_size
        )
        return ctx_size

    def _make_llm_chain(self, prompt_text: str) -> RunnableLambda:
        """
        Each call returns:
          {
            "summary": <str>,
            "text_tokens": int,
            "prompt_tokens": int,
            "total_needed": int,
            "used_ctx": int
          }
        """
        def chain_func(inputs: Dict[str, str]) -> Dict[str, str or int]:
            text = inputs["context"]

            # Count text tokens
            text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            # Count prompt tokens
            prompt_tokens = len(tokenizer.encode(prompt_text.replace("{context}", text), add_special_tokens=False))
            total_needed = prompt_tokens  # total prompt tokens

            # (If dynamic) re-init LLM with final needed tokens
            used_ctx = self._init_model_for_tokens(total_needed)

            # Actually call the LLM
            try:
                resp = self.model.invoke([HumanMessage(content=prompt_text.replace("{context}", text))])
                summary = resp.content.strip()
            except Exception as e:
                logger.error(f"LLM error: {e}")
                summary = ""

            return {
                "summary": summary,
                "text_tokens": text_tokens,
                "prompt_tokens": prompt_tokens,
                "total_needed": total_needed,
                "used_ctx": used_ctx
            }
        return RunnableLambda(chain_func)

    ###########################################################################
    # Single-step combined
    ###########################################################################
    def summarize_combined_singlestep(self, text: str) -> Dict[str, any]:
        if not self.runnable_combined:
            logger.warning("No combined prompt loaded.")
            return {"summary": ""}
        result = self.runnable_combined.invoke({"context": text})
        result["summary"] = enforce_format(result["summary"], ALL_FIELDS)
        return result

    ###########################################################################
    # Two-step combined
    ###########################################################################
    def summarize_combined_twostep(self, text: str) -> Dict[str, any]:
        out_dict = {
            "summary": "",
            # We'll store step1.. step2.. then final merges
            "extr_text_tokens": None,
            "extr_prompt_tokens": None,
            "extr_total_needed": None,
            "extr_used_ctx": None,
            "cot_text_tokens": None,
            "cot_prompt_tokens": None,
            "cot_total_needed": None,
            "cot_used_ctx": None
        }

        if not (self.runnable_extraction and self.runnable_cot):
            # fallback to single-step
            single_res = self.summarize_combined_singlestep(text)
            out_dict["summary"] = single_res["summary"]
            return out_dict

        # Step 1 => extraction
        ex_res = self.runnable_extraction.invoke({"context": text})
        # Step 2 => CoT
        cot_res = self.runnable_cot.invoke({"context": text})
        merged = ex_res["summary"] + "\n" + cot_res["summary"]
        final = enforce_format(merged, ALL_FIELDS)
        # Store step stats
        out_dict["summary"] = final
        out_dict["extr_text_tokens"]   = ex_res["text_tokens"]
        out_dict["extr_prompt_tokens"] = ex_res["prompt_tokens"]
        out_dict["extr_total_needed"]  = ex_res["total_needed"]
        out_dict["extr_used_ctx"]      = ex_res["used_ctx"]
        out_dict["cot_text_tokens"]    = cot_res["text_tokens"]
        out_dict["cot_prompt_tokens"]  = cot_res["prompt_tokens"]
        out_dict["cot_total_needed"]   = cot_res["total_needed"]
        out_dict["cot_used_ctx"]       = cot_res["used_ctx"]
        return out_dict

    ###########################################################################
    # Single-step separate
    ###########################################################################
    def summarize_separate_singlestep(self, path_text: str, cons_text: str) -> Dict[str, any]:
        """
        We'll do single-step on path_text => path_res
        single-step on cons_text => cons_res
        Then merge.
        We'll store columns for path_..., cons_...
        """
        path_res = self.summarize_combined_singlestep(path_text)
        cons_res = self.summarize_combined_singlestep(cons_text)
        merged = merge_summaries(path_res["summary"], cons_res["summary"], ALL_FIELDS)
        out_dict = {
            "summary": merged,
            "path_text_tokens": path_res["text_tokens"],
            "path_prompt_tokens": path_res["prompt_tokens"],
            "path_total_needed": path_res["total_needed"],
            "path_used_ctx": path_res["used_ctx"],

            "cons_text_tokens": cons_res["text_tokens"],
            "cons_prompt_tokens": cons_res["prompt_tokens"],
            "cons_total_needed": cons_res["total_needed"],
            "cons_used_ctx": cons_res["used_ctx"]
        }
        return out_dict

    ###########################################################################
    # Two-step separate
    ###########################################################################
    def summarize_separate_twostep(self, path_text: str, cons_text: str) -> Dict[str, any]:
        """
        path => two-step, cons => two-step, then merge
        we'll store path_extr_..., path_cot_..., cons_extr_..., cons_cot_...
        """
        path_out = self.summarize_combined_twostep(path_text)
        cons_out = self.summarize_combined_twostep(cons_text)
        final = merge_summaries(path_out["summary"], cons_out["summary"], ALL_FIELDS)
        out_dict = {
            "summary": final,
            # path step stats
            "path_extr_text_tokens": path_out.get("extr_text_tokens"),
            "path_extr_prompt_tokens": path_out.get("extr_prompt_tokens"),
            "path_extr_total_needed": path_out.get("extr_total_needed"),
            "path_extr_used_ctx": path_out.get("extr_used_ctx"),

            "path_cot_text_tokens": path_out.get("cot_text_tokens"),
            "path_cot_prompt_tokens": path_out.get("cot_prompt_tokens"),
            "path_cot_total_needed": path_out.get("cot_total_needed"),
            "path_cot_used_ctx": path_out.get("cot_used_ctx"),

            # cons step stats
            "cons_extr_text_tokens": cons_out.get("extr_text_tokens"),
            "cons_extr_prompt_tokens": cons_out.get("extr_prompt_tokens"),
            "cons_extr_total_needed": cons_out.get("extr_total_needed"),
            "cons_extr_used_ctx": cons_out.get("extr_used_ctx"),

            "cons_cot_text_tokens": cons_out.get("cot_text_tokens"),
            "cons_cot_prompt_tokens": cons_out.get("cot_prompt_tokens"),
            "cons_cot_total_needed": cons_out.get("cot_total_needed"),
            "cons_cot_used_ctx": cons_out.get("cot_used_ctx")
        }
        return out_dict

    ###########################################################################
    # Summarize one patient
    ###########################################################################
    def summarize_one_patient(self, path_text: str, cons_text: str) -> Dict[str, any]:
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
            logger.error(f"Invalid experiment_mode={mode}")
            return {"summary": ""}

    ###########################################################################
    # Precompute tokens only
    ###########################################################################
    def precompute_tokens(self, input_dir: str, output_dir: str):
        records = []
        # combined
        cdir = os.path.join(input_dir, "PathConsCombined")
        if os.path.isdir(cdir):
            for root, _, fs in os.walk(cdir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        pid = os.path.splitext(fname)[0]
                        with open(fp, 'r', encoding='utf-8') as f:
                            text = f.read()
                        n_chars = len(text)
                        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                        records.append({
                            "report_type": "combined",
                            "patient_id": pid,
                            "file": fname,
                            "num_chars": n_chars,
                            "num_tokens": n_tokens
                        })

        # pathology
        pdir = os.path.join(input_dir, "PathologyReports")
        if os.path.isdir(pdir):
            for root, _, fs in os.walk(pdir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        pid = os.path.splitext(fname)[0]
                        with open(fp, 'r', encoding='utf-8') as f:
                            text = f.read()
                        n_chars = len(text)
                        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                        records.append({
                            "report_type": "pathology",
                            "patient_id": pid,
                            "file": fname,
                            "num_chars": n_chars,
                            "num_tokens": n_tokens
                        })

        # consult
        cndir = os.path.join(input_dir, "ConsultRedacted")
        if os.path.isdir(cndir):
            for root, _, fs in os.walk(cndir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        pid = os.path.splitext(fname)[0]
                        with open(fp, 'r', encoding='utf-8') as f:
                            text = f.read()
                        n_chars = len(text)
                        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                        records.append({
                            "report_type": "consult",
                            "patient_id": pid,
                            "file": fname,
                            "num_chars": n_chars,
                            "num_tokens": n_tokens
                        })

        if records:
            df = pd.DataFrame(records)
            outpath = os.path.join(output_dir, "precompute_tokens.csv")
            df.to_csv(outpath, index=False)
            logger.info(f"precompute_tokens => {outpath}")
        else:
            logger.warning("No .txt found for precompute.")

    ###########################################################################
    # Normal Summarization
    ###########################################################################
    def process_reports(
        self,
        input_dir: str,
        output_dir: str,
        case_ids: List[str],
        single: bool=False
    ):
        """
        case_ids can be a list (including the user-specified or from file).
        If case_ids is nonempty, we only process those patients.
        If single, we pick one random from the entire set.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Subfolder naming
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

        time_records = []

        #######################################################################
        # (1) Combined
        #######################################################################
        if combined_needed:
            comb_dir = os.path.join(input_dir, "PathConsCombined")
            if not os.path.isdir(comb_dir):
                logger.error(f"Missing PathConsCombined => {comb_dir}")
                return

            all_files = []
            for root, _, fs in os.walk(comb_dir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        pid = os.path.splitext(fname)[0]
                        if case_ids and pid not in case_ids:
                            continue
                        all_files.append(os.path.join(root, fname))

            if single and all_files:
                all_files = [random.choice(all_files)]

            for fp in all_files:
                fname = os.path.basename(fp)
                pid = os.path.splitext(fname)[0]
                logger.info(f"[Mode {self.experiment_mode}] Summarizing => {pid}")

                with open(fp, 'r', encoding='utf-8') as f:
                    text = f.read()
                in_chars = len(text)
                in_tokens = len(tokenizer.encode(text, add_special_tokens=False))

                st = time.time()
                out_dict = self.summarize_one_patient(text, "")
                et = time.time()
                ms = int(round((et - st)*1000))

                # Build row
                row = {
                    "file": fname,
                    "report_type": subfolder,
                    "process_time_ms": ms,
                    "num_input_characters": in_chars,
                    "num_input_tokens": in_tokens,
                    # "patient_id": pid
                }
                # Also merge columns from out_dict
                for k,v in out_dict.items():
                    if k == "summary":
                        continue
                    row[k] = v
                summary = out_dict["summary"]

                if not summary:
                    logger.warning(f"No summary => {pid}")
                    continue

                # Save
                text_dir = os.path.join(output_dir, "text_summaries", subfolder, pid)
                emb_dir  = os.path.join(output_dir, "embeddings", subfolder, pid)
                os.makedirs(text_dir, exist_ok=True)
                os.makedirs(emb_dir, exist_ok=True)
                with open(os.path.join(text_dir, "path_consult_reports_summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(summary)
                emb = self.embeddings.embed_documents([summary])[0]
                with open(os.path.join(emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

                time_records.append(row)

        #######################################################################
        # (2) Separate
        #######################################################################
        else:
            pdir = os.path.join(input_dir, "PathologyReports")
            cdir = os.path.join(input_dir, "ConsultRedacted")
            if not (os.path.isdir(pdir) or os.path.isdir(cdir)):
                logger.error("Missing PathologyReports or ConsultRedacted.")
                return

            path_map = {}
            if os.path.isdir(pdir):
                for root, _, fs in os.walk(pdir):
                    for fname in fs:
                        if fname.endswith(".txt"):
                            pid = os.path.splitext(fname)[0]
                            path_map[pid] = os.path.join(root, fname)
            cons_map = {}
            if os.path.isdir(cdir):
                for root, _, fs in os.walk(cdir):
                    for fname in fs:
                        if fname.endswith(".txt"):
                            pid = os.path.splitext(fname)[0]
                            cons_map[pid] = os.path.join(root, fname)

            all_pids = set(path_map.keys()) | set(cons_map.keys())
            if case_ids:
                all_pids = all_pids.intersection(case_ids)
            if single and all_pids:
                chosen = random.choice(list(all_pids))
                all_pids = {chosen}

            for pid in all_pids:
                logger.info(f"[Mode {self.experiment_mode}] Summarizing => {pid}")
                path_text = ""
                if pid in path_map:
                    with open(path_map[pid], 'r', encoding='utf-8') as f:
                        path_text = f.read()
                cons_text = ""
                if pid in cons_map:
                    with open(cons_map[pid], 'r', encoding='utf-8') as f:
                        cons_text = f.read()

                combined = path_text + "\n\n" + cons_text
                in_chars = len(combined)
                in_tokens = len(tokenizer.encode(combined, add_special_tokens=False))

                st = time.time()
                out_dict = self.summarize_one_patient(path_text, cons_text)
                et = time.time()
                ms = int(round((et - st)*1000))

                row = {
                    "file": f"{pid}.txt",
                    "report_type": subfolder,
                    "process_time_ms": ms,
                    "num_input_characters": in_chars,
                    "num_input_tokens": in_tokens,
                    "patient_id": pid
                }
                for k,v in out_dict.items():
                    if k == "summary":
                        continue
                    row[k] = v
                summary = out_dict["summary"]

                if not summary:
                    logger.warning(f"No summary => {pid}")
                    continue

                text_dir = os.path.join(output_dir, "text_summaries", subfolder, pid)
                emb_dir  = os.path.join(output_dir, "embeddings", subfolder, pid)
                os.makedirs(text_dir, exist_ok=True)
                os.makedirs(emb_dir, exist_ok=True)
                with open(os.path.join(text_dir, "path_consult_reports_summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(summary)
                emb = self.embeddings.embed_documents([summary])[0]
                with open(os.path.join(emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

                time_records.append(row)

        #######################################################################
        # Save the DF
        #######################################################################
        if time_records:
            df = pd.DataFrame(time_records)
            csv_path = os.path.join(output_dir, "processing_times.csv")
            if os.path.isfile(csv_path):
                old_df = pd.read_csv(csv_path)
                # union columns
                df_all_cols = pd.concat([old_df, df], ignore_index=True, sort=False)
                df_all_cols.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, index=False)
            logger.info(f"Saved timing => {csv_path}")
        else:
            logger.warning("No records to save.")

###############################################################################
# 5. CLI
###############################################################################
def main():
    parser = argparse.ArgumentParser("HNC Summarizer Agent7: multi-step tokens + dynamic ctx + multiple caseIDs")
    parser.add_argument("--prompts_dir", required=True, help="Folder with JSON prompt files.")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend.")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature.")
    parser.add_argument("--embedding_model", default="ollama", choices=["ollama","openai","google"], help="Emb model.")
    parser.add_argument("--local_model", default="llama3.3:latest", help="Local model name if model_type=local.")
    parser.add_argument("--experiment_mode", type=int, default=1, help="Modes: 1..4.")
    parser.add_argument("--input_dir", required=True, help="Parent dir with PathologyReports, PathConsCombined, etc.")
    parser.add_argument("--output_dir", required=True, help="Output location.")
    parser.add_argument("--dynamic_ctx_window", action="store_true", help="Auto-calc num_ctx per call.")
    parser.add_argument("--default_ctx", type=int, default=2048, help="Default context if needed < this.")
    parser.add_argument("--margin", type=int, default=500, help="Tokens margin if needed > default_ctx.")
    parser.add_argument("--ollama_context_size", type=int, default=2048,
                        help="Fallback num_ctx if not dynamic_ctx_window.")
    parser.add_argument("--case_id", default="", help="Single case. (One ID.)")
    parser.add_argument("--case_ids", default="", help="Comma-separated multiple IDs.")
    parser.add_argument("--single", action="store_true", help="If set, pick random from the matching set.")
    parser.add_argument("--precompute_tokens", action="store_true",
                        help="If set, only gather token counts for .txt and store in precompute_tokens.csv, no LLM calls.")
    args = parser.parse_args()

    # Build Summarizer
    summarizer = ReportSummarizer(
        prompts_dir=args.prompts_dir,
        model_type=args.model_type,
        temperature=args.temperature,
        embedding_model=args.embedding_model,
        local_model=args.local_model,
        experiment_mode=args.experiment_mode,
        dynamic_ctx_window=args.dynamic_ctx_window,
        default_ctx=args.default_ctx,
        margin=args.margin,
        ollama_context_size=args.ollama_context_size
    )

    # If precompute => do that and exit
    if args.precompute_tokens:
        os.makedirs(args.output_dir, exist_ok=True)
        summarizer.precompute_tokens(args.input_dir, args.output_dir)
        return

    # Gather caseIDs if any
    final_case_ids = set()
    if args.case_id.strip():
        final_case_ids.add(args.case_id.strip())
    if args.case_ids.strip():
        for cid in args.case_ids.split(","):
            cid = cid.strip()
            if cid:
                final_case_ids.add(cid)

    summarizer.process_reports(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        case_ids=list(final_case_ids),
        single=args.single
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

# USING 1130120 as a test case id since it had a very long combined report (which didn't work well with hnc_reports_agent4.py)

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

# NOW DO THE PRECOMPUTATION OF THE TOKENS FOR EXPERIMENT 2 (COMBINED + TWO-STEP SUMMARIES)
# GET A LIST OF CASE IDS WHOSE TOTAL TOKENS NEEDED WERE BELOW VERSUS ABOVE 2048 THE CHATOLLAMA DEFAULT 
# THEN MAKE SURE THE CASE_ID ARGUMENT ACN PROCESS THIS .CSV LIST OF THEM ONLY IN EXP14. 

# FOR THE COLLECTION OF FINAL RESULTS THEY CAN BE MERGED FROM EXP13 FOR THE ONES BELOW 2048 AND THE ONES ABOVE 2048 FROM EXP14.
# PERHAPS SAVE THIS COMBINED FINAL UNIQUE 882 CASES INTO ANOTHER FOLDER? FOR PAPER 2 PROCESSING 

# REDO THE 50 CASE SELECTIONS FROM AN ACTUAL RANGE OF THE INPUT TOKENS BASED ON EXP MODE2 INSTEAD OF THE NUMBER OF NOT INFERRED! 
# DISCUSS WITH LAYA AND FARHOOD AND BE DONE!