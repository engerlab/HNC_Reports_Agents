#!/usr/bin/env python3
"""
HNC Summarizer with 4 Modes & Automatic Precompute Tokens (One CSV per Mode)

This script processes head-and-neck cancer (HNC) patient reports by extracting a
30-line structured summary (fields like Pathological_TNM, Clinical_TNM, etc.).
It supports four experiment modes:

  Mode 1 => Combined text + Single-step prompt
  Mode 2 => Combined text + Two-step prompt (extraction + CoT)
  Mode 3 => Separate texts (Pathology + Consultation) + Single-step prompt
  Mode 4 => Separate texts (Pathology + Consultation) + Two-step prompt

When --precompute_tokens is provided, the script:
  1) Loops over modes 1..4.
  2) For each mode, it calculates how many tokens each LLM call would need.
  3) Saves the results in precompute_tokens_modeX.csv in the output directory.
Columns differ by mode:
- Mode 1 (Combined single-step):
    file,report_type,num_input_characters,text_tokens,prompt_instr_tokens,total_needed
- Mode 2 (Combined two-step):
    file,report_type,num_input_characters,
    extr_text_tokens,extr_prompt_instr_tokens,extr_total_needed,
    cot_text_tokens,cot_prompt_instr_tokens,cot_total_needed
- Mode 3 (Separate single-step):
    Each subcall is either path or consult => one row each
    file,report_type,subcall ("path" or "consult"),
    text_tokens,prompt_instr_tokens,total_needed
- Mode 4 (Separate two-step):
    Each subcall is either path-extraction, path-cot, consult-extraction, consult-cot => up to 4 rows
    with columns like extr_text_tokens, extr_prompt_instr_tokens, etc.

When not using --precompute_tokens, the script runs normal summarization for the chosen mode only.

New Features (added):
  - You can set a fixed context window (e.g. 4096) by *not* passing --dynamic_ctx_window and 
    specifying --ollama_context_size 4096.
  - Subset filtering for mode=2 or other modes using a precompute CSV:
     --subset_precompute_csv PATH
     --only_above INT
    This filters those rows where (extr_total_needed>only_above) or (cot_total_needed>only_above),
    extracts the case ID (file minus ".txt"), and runs only those.

Usage example for subset & fixed context:
  python hnc_reports_agent7.py \
    --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
    --model_type local \
    --temperature 0.8 \
    --local_model "llama3.3:latest" \
    --embedding_model ollama \
    --input_dir "/media/yujing/One Touch3/HNC_Reports" \
    --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPrompt37" \
    --experiment_mode 2 \
    --ollama_context_size 4096 \
    --subset_precompute_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode2.csv" \
    --only_above 1500
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

# For token counting
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
except:
    raise ValueError("Cannot load 'meta-llama/Llama-3.1-70B' tokenizer. Update if needed.")

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
    "Treatment_Recommendations",
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
# 2. Format Enforcer
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
        dynamic_ctx_window: bool = False,
        default_ctx: int = 2048,
        margin: int = 500,
        ollama_context_size: int = 2048
    ):
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

        # Load prompts for summarization
        self.prompt_combined   = self._load_prompt("prompt_path_consult_reports_combined.json")
        self.prompt_extraction = self._load_prompt("prompt_path_consult_reports_extraction.json")
        self.prompt_cot        = self._load_prompt("prompt_path_consult_reports_cot.json")

        # If not dynamic, we init one local model
        if self.model_type == "local" and not self.dynamic_ctx_window:
            self.model = ChatOllama(
                model=self.local_model,
                temperature=self.temperature,
                num_ctx=self.ollama_context_size
            )
        elif self.model_type == "local" and self.dynamic_ctx_window:
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
            logging.warning(f"Unknown embedding_model={self.embedding_model}, defaulting to Ollama.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build chain
        self.runnable_combined   = self._make_llm_chain(self.prompt_combined)    if self.prompt_combined    else None
        self.runnable_extraction = self._make_llm_chain(self.prompt_extraction)  if self.prompt_extraction  else None
        self.runnable_cot        = self._make_llm_chain(self.prompt_cot)         if self.prompt_cot         else None


    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.prompts_dir, filename)
        if not os.path.isfile(path):
            logger.warning(f"No prompt file found: {path}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "\n".join(data.get("prompts", []))

    def _init_model_for_tokens(self, needed_tokens: int) -> int:
        if self.model_type != "local":
            return 0
        if not self.dynamic_ctx_window:
            return self.ollama_context_size

        if needed_tokens < self.default_ctx:
            ctx = self.default_ctx
        else:
            ctx = needed_tokens + self.margin
        logger.debug(f"[Dynamic ctx] needed={needed_tokens}, used={ctx}")
        self.model = ChatOllama(
            model=self.local_model,
            temperature=self.temperature,
            num_ctx=ctx
        )
        return ctx

    def _make_llm_chain(self, prompt_text: str) -> RunnableLambda:
        def chain_func(inputs: Dict[str, str]) -> Dict[str, any]:
            user_text = inputs["context"]
            instructions = prompt_text

            text_tokens = len(tokenizer.encode(user_text, add_special_tokens=False))
            prompt_noctx = instructions.replace("{context}", "")
            prompt_instr_tokens = len(tokenizer.encode(prompt_noctx, add_special_tokens=False))

            total_needed = text_tokens + prompt_instr_tokens
            used_ctx = self._init_model_for_tokens(total_needed)

            final_prompt = instructions.replace("{context}", user_text)
            summary = ""
            try:
                resp = self.model.invoke([HumanMessage(content=final_prompt)])
                summary = resp.content.strip()
            except Exception as e:
                logger.error(f"LLM error: {e}")

            return {
                "summary": summary,
                "text_tokens": text_tokens,
                "prompt_instr_tokens": prompt_instr_tokens,
                "total_needed": total_needed,
                "used_ctx": used_ctx
            }
        return RunnableLambda(chain_func)

    ###########################################################################
    # Summaries
    ###########################################################################
    def summarize_combined_singlestep(self, text: str) -> Dict[str, any]:
        if not self.runnable_combined:
            logger.warning("No combined prompt loaded.")
            return {"summary": ""}
        result = self.runnable_combined.invoke({"context": text})
        final_summary = enforce_format(result["summary"], ALL_FIELDS)
        result["summary"] = final_summary
        return result

    def summarize_combined_twostep(self, text: str) -> Dict[str, any]:
        out_dict = {"summary": ""}
        if not (self.runnable_extraction and self.runnable_cot):
            single_res = self.summarize_combined_singlestep(text)
            out_dict["summary"] = single_res["summary"]
            return out_dict

        ex_res = self.runnable_extraction.invoke({"context": text})
        cot_res = self.runnable_cot.invoke({"context": text})
        merged = ex_res["summary"] + "\n" + cot_res["summary"]
        final = enforce_format(merged, ALL_FIELDS)
        out_dict["summary"] = final

        out_dict["extr_text_tokens"] = ex_res["text_tokens"]
        out_dict["extr_prompt_instr_tokens"] = ex_res["prompt_instr_tokens"]
        out_dict["extr_total_needed"] = ex_res["total_needed"]
        out_dict["extr_used_ctx"] = ex_res["used_ctx"]

        out_dict["cot_text_tokens"] = cot_res["text_tokens"]
        out_dict["cot_prompt_instr_tokens"] = cot_res["prompt_instr_tokens"]
        out_dict["cot_total_needed"] = cot_res["total_needed"]
        out_dict["cot_used_ctx"] = cot_res["used_ctx"]
        return out_dict

    def summarize_separate_singlestep(self, path_text: str, cons_text: str) -> Dict[str, any]:
        p_res = self.summarize_combined_singlestep(path_text)
        c_res = self.summarize_combined_singlestep(cons_text)
        merged = merge_summaries(p_res["summary"], c_res["summary"], ALL_FIELDS)
        return {
            "summary": merged,
            "path_text_tokens": p_res["text_tokens"],
            "path_prompt_instr_tokens": p_res["prompt_instr_tokens"],
            "path_total_needed": p_res["total_needed"],
            "path_used_ctx": p_res["used_ctx"],

            "cons_text_tokens": c_res["text_tokens"],
            "cons_prompt_instr_tokens": c_res["prompt_instr_tokens"],
            "cons_total_needed": c_res["total_needed"],
            "cons_used_ctx": c_res["used_ctx"]
        }

    def summarize_separate_twostep(self, path_text: str, cons_text: str) -> Dict[str, any]:
        p_out = self.summarize_combined_twostep(path_text)
        c_out = self.summarize_combined_twostep(cons_text)
        final = merge_summaries(p_out["summary"], c_out["summary"], ALL_FIELDS)

        return {
            "summary": final,

            "path_extr_text_tokens": p_out.get("extr_text_tokens"),
            "path_extr_prompt_instr_tokens": p_out.get("extr_prompt_instr_tokens"),
            "path_extr_total_needed": p_out.get("extr_total_needed"),
            "path_extr_used_ctx": p_out.get("extr_used_ctx"),

            "path_cot_text_tokens": p_out.get("cot_text_tokens"),
            "path_cot_prompt_instr_tokens": p_out.get("cot_prompt_instr_tokens"),
            "path_cot_total_needed": p_out.get("cot_total_needed"),
            "path_cot_used_ctx": p_out.get("cot_used_ctx"),

            "cons_extr_text_tokens": c_out.get("extr_text_tokens"),
            "cons_extr_prompt_instr_tokens": c_out.get("extr_prompt_instr_tokens"),
            "cons_extr_total_needed": c_out.get("extr_total_needed"),
            "cons_extr_used_ctx": c_out.get("extr_used_ctx"),

            "cons_cot_text_tokens": c_out.get("cot_text_tokens"),
            "cons_cot_prompt_instr_tokens": c_out.get("cot_prompt_instr_tokens"),
            "cons_cot_total_needed": c_out.get("cot_total_needed"),
            "cons_cot_used_ctx": c_out.get("cot_used_ctx")
        }

    def summarize_one_patient(self, path_text: str, cons_text: str) -> Dict[str, any]:
        if self.experiment_mode == 1:
            return self.summarize_combined_singlestep(path_text)
        elif self.experiment_mode == 2:
            return self.summarize_combined_twostep(path_text)
        elif self.experiment_mode == 3:
            return self.summarize_separate_singlestep(path_text, cons_text)
        elif self.experiment_mode == 4:
            return self.summarize_separate_twostep(path_text, cons_text)
        else:
            logger.error(f"Invalid experiment_mode={self.experiment_mode}")
            return {"summary": ""}

    ###########################################################################
    # The new approach: produce 4 CSVs (one for each mode) automatically
    ###########################################################################
    def precompute_tokens_for_all_modes(self, input_dir: str, output_dir: str):
        """
        We'll produce 4 CSVs:
          precompute_tokens_mode1.csv
          precompute_tokens_mode2.csv
          precompute_tokens_mode3.csv
          precompute_tokens_mode4.csv
        each enumerates the token usage for each .txt (and subcalls if separate mode)
        without actually calling the LLM.
        """

        # We'll define 4 methods
        # Then we call each method => output a separate DF => write to precompute_tokens_modeX.csv
        for mode in [1,2,3,4]:
            logger.info(f"[Precompute] Generating CSV for mode={mode}")
            df_result = self._precompute_tokens_for_mode(mode, input_dir)
            if df_result.empty:
                logger.warning(f"No data produced for mode {mode}.")
                continue
            outpath = os.path.join(output_dir, f"precompute_tokens_mode{mode}.csv")
            df_result.to_csv(outpath, index=False)
            logger.info(f"Saved => {outpath}")

    def _precompute_tokens_for_mode(self, mode: int, input_dir: str) -> pd.DataFrame:
        """
        Each mode does a certain approach:
         - Mode 1 => Combined single-step
         - Mode 2 => Combined two-step
         - Mode 3 => Separate single-step => 2 subcalls per patient if both exist
         - Mode 4 => Separate two-step => up to 4 subcalls per patient
        We'll gather rows with relevant columns.

        For each mode, we scan the relevant subfolders:
          mode1,2 => PathConsCombined
          mode3,4 => PathologyReports + ConsultRedacted
        """
        records = []

        # load needed prompts
        combined_prompt = self.prompt_combined or ""
        extraction_prompt = self.prompt_extraction or ""
        cot_prompt = self.prompt_cot or ""

        # remove {context}
        comb_noctx = combined_prompt.replace("{context}", "")
        comb_instr_tokens = len(tokenizer.encode(comb_noctx, add_special_tokens=False)) if comb_noctx else 0

        extr_noctx = extraction_prompt.replace("{context}", "")
        extr_instr_tokens = len(tokenizer.encode(extr_noctx, add_special_tokens=False)) if extr_noctx else 0

        cot_noctx = cot_prompt.replace("{context}", "")
        cot_instr_tokens = len(tokenizer.encode(cot_noctx, add_special_tokens=False)) if cot_noctx else 0

        def path_cons_combined(folder):
            # gather all .txt
            fullp = os.path.join(input_dir, folder)
            if not os.path.isdir(fullp):
                return []
            result = []
            for root, _, fs in os.walk(fullp):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        with open(fp, 'r', encoding='utf-8') as f:
                            text = f.read()
                        result.append((fname, text))
            return result

        if mode in [1,2]:
            # read from PathConsCombined
            combined_list = path_cons_combined("PathConsCombined")
            for (fname, text) in combined_list:
                row_base = {
                    "file": fname,
                    "report_type": "combined",
                    "num_input_characters": len(text)
                }
                if mode == 1:
                    # single-step => text_tokens + comb_instr_tokens => total_needed
                    toks_text = len(tokenizer.encode(text, add_special_tokens=False))
                    row_base["text_tokens"] = toks_text
                    row_base["prompt_instr_tokens"] = comb_instr_tokens
                    row_base["total_needed"] = toks_text + comb_instr_tokens
                    records.append(row_base)
                else:
                    # mode 2 => two-step => extraction + CoT
                    toks_text = len(tokenizer.encode(text, add_special_tokens=False))

                    # create row with all columns
                    row_base["extr_text_tokens"] = toks_text
                    row_base["extr_prompt_instr_tokens"] = extr_instr_tokens
                    row_base["extr_total_needed"] = toks_text + extr_instr_tokens

                    row_base["cot_text_tokens"] = toks_text
                    row_base["cot_prompt_instr_tokens"] = cot_instr_tokens
                    row_base["cot_total_needed"] = toks_text + cot_instr_tokens

                    records.append(row_base)

        else:
            # mode 3 or 4 => separate => PathologyReports, ConsultRedacted
            path_map = {}
            pdir = os.path.join(input_dir, "PathologyReports")
            if os.path.isdir(pdir):
                for root, _, fs in os.walk(pdir):
                    for fname in fs:
                        if fname.endswith(".txt"):
                            fp = os.path.join(root, fname)
                            with open(fp, 'r', encoding='utf-8') as f:
                                text = f.read()
                            pid = os.path.splitext(fname)[0]
                            path_map[pid] = (fname, text)

            cons_map = {}
            cdir = os.path.join(input_dir, "ConsultRedacted")
            if os.path.isdir(cdir):
                for root, _, fs in os.walk(cdir):
                    for fname in fs:
                        if fname.endswith(".txt"):
                            fp = os.path.join(root, fname)
                            with open(fp, 'r', encoding='utf-8') as f:
                                text = f.read()
                            pid = os.path.splitext(fname)[0]
                            cons_map[pid] = (fname, text)

            all_pids = set(path_map.keys()) | set(cons_map.keys())
            for pid in all_pids:
                path_data = path_map.get(pid, None)
                cons_data = cons_map.get(pid, None)

                if mode == 3:
                    # single-step => path_text + combined prompt => total_needed
                    if path_data:
                        fname, text = path_data
                        row_p = {
                            "file": fname,
                            "report_type": "pathology",
                            "num_input_characters": len(text),
                            "subcall": "path"
                        }
                        tpath = len(tokenizer.encode(text, add_special_tokens=False))
                        row_p["text_tokens"] = tpath
                        row_p["prompt_instr_tokens"] = comb_instr_tokens
                        row_p["total_needed"] = tpath + comb_instr_tokens
                        records.append(row_p)

                    if cons_data:
                        fname, text = cons_data
                        row_c = {
                            "file": fname,
                            "report_type": "consult",
                            "num_input_characters": len(text),
                            "subcall": "consult"
                        }
                        tcons = len(tokenizer.encode(text, add_special_tokens=False))
                        row_c["text_tokens"] = tcons
                        row_c["prompt_instr_tokens"] = comb_instr_tokens
                        row_c["total_needed"] = tcons + comb_instr_tokens
                        records.append(row_c)

                else:
                    # mode == 4 => separate two-step => path_extr, path_cot, cons_extr, cons_cot
                    if path_data:
                        fname, text = path_data
                        tpath = len(tokenizer.encode(text, add_special_tokens=False))
                        # extraction step
                        row_pe = {
                            "file": fname,
                            "report_type": "pathology",
                            "subcall": "path_extraction",
                            "num_input_characters": len(text),
                            "extr_text_tokens": tpath,
                            "extr_prompt_instr_tokens": extr_instr_tokens,
                            "extr_total_needed": tpath + extr_instr_tokens
                        }
                        records.append(row_pe)
                        # CoT step
                        row_pc = {
                            "file": fname,
                            "report_type": "pathology",
                            "subcall": "path_cot",
                            "num_input_characters": len(text),
                            "cot_text_tokens": tpath,
                            "cot_prompt_instr_tokens": cot_instr_tokens,
                            "cot_total_needed": tpath + cot_instr_tokens
                        }
                        records.append(row_pc)

                    if cons_data:
                        fname, text = cons_data
                        tcons = len(tokenizer.encode(text, add_special_tokens=False))
                        # extraction
                        row_ce = {
                            "file": fname,
                            "report_type": "consult",
                            "subcall": "cons_extraction",
                            "num_input_characters": len(text),
                            "extr_text_tokens": tcons,
                            "extr_prompt_instr_tokens": extr_instr_tokens,
                            "extr_total_needed": tcons + extr_instr_tokens
                        }
                        records.append(row_ce)
                        # cot
                        row_cc = {
                            "file": fname,
                            "report_type": "consult",
                            "subcall": "cons_cot",
                            "num_input_characters": len(text),
                            "cot_text_tokens": tcons,
                            "cot_prompt_instr_tokens": cot_instr_tokens,
                            "cot_total_needed": tcons + cot_instr_tokens
                        }
                        records.append(row_cc)

        return pd.DataFrame(records)

    ###########################################################################
    # Normal method => precompute_tokens
    ###########################################################################
    def precompute_tokens(self, input_dir: str, output_dir: str):
        """
        We'll produce 4 separate CSV files in output_dir:
          precompute_tokens_mode1.csv
          precompute_tokens_mode2.csv
          precompute_tokens_mode3.csv
          precompute_tokens_mode4.csv
        capturing the tokens needed for each of the 4 experiment modes.
        """
        os.makedirs(output_dir, exist_ok=True)
        for mode in [1,2,3,4]:
            df = self._precompute_tokens_for_mode(mode, input_dir)
            if df.empty:
                logger.warning(f"No data for mode {mode}, skipping.")
                continue
            outfn = f"precompute_tokens_mode{mode}.csv"
            outpath = os.path.join(output_dir, outfn)
            df.to_csv(outpath, index=False)
            logger.info(f"[Precompute] Saved => {outpath}")

    ###########################################################################
    # Normal Summarization => process_reports
    ###########################################################################
    def process_reports(
        self,
        input_dir: str,
        output_dir: str,
        case_ids: List[str],
        single: bool=False
    ):
        # Summarize for the chosen experiment_mode (1..4)
        os.makedirs(output_dir, exist_ok=True)

        # subfolder naming
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
        if combined_needed:
            # read from PathConsCombined
            cdir = os.path.join(input_dir, "PathConsCombined")
            if not os.path.isdir(cdir):
                logger.error("Missing PathConsCombined")
                return
            all_files = []
            for root, _, fs in os.walk(cdir):
                for fname in fs:
                    if fname.endswith(".txt"):
                        pid = os.path.splitext(fname)[0]
                        if case_ids and (pid not in case_ids):
                            continue
                        all_files.append(os.path.join(root, fname))
            if single and all_files:
                all_files = [random.choice(all_files)]

            for fp in all_files:
                fname = os.path.basename(fp)
                with open(fp, 'r', encoding='utf-8') as f:
                    text = f.read()
                num_chars = len(text)

                st = time.time()
                out_dict = self.summarize_one_patient(text, "")
                et = time.time()
                ms = int(round((et - st)*1000))

                row = {
                    "file": fname,
                    "report_type": subfolder,
                    "process_time_ms": ms,
                    "num_input_characters": num_chars,
                }
                for k,v in out_dict.items():
                    if k == "summary":
                        continue
                    row[k] = v

                if not out_dict["summary"]:
                    logger.warning(f"No summary for {fname}")
                    continue

                pid = os.path.splitext(fname)[0]
                txt_dir = os.path.join(output_dir, "text_summaries", subfolder, pid)
                emb_dir = os.path.join(output_dir, "embeddings", subfolder, pid)
                os.makedirs(txt_dir, exist_ok=True)
                os.makedirs(emb_dir, exist_ok=True)

                with open(os.path.join(txt_dir, "path_consult_reports_summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(out_dict["summary"])
                emb = self.embeddings.embed_documents([out_dict["summary"]])[0]
                with open(os.path.join(emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

                time_records.append(row)

        else:
            # separate => Pathology + Consultation
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
                path_text = ""
                if pid in path_map:
                    with open(path_map[pid], 'r', encoding='utf-8') as f:
                        path_text = f.read()
                cons_text = ""
                if pid in cons_map:
                    with open(cons_map[pid], 'r', encoding='utf-8') as f:
                        cons_text = f.read()

                combined_input = path_text + "\n\n" + cons_text
                n_chars = len(combined_input)

                st = time.time()
                out_dict = self.summarize_one_patient(path_text, cons_text)
                et = time.time()
                ms = int(round((et - st)*1000))

                row = {
                    "file": f"{pid}.txt",
                    "report_type": subfolder,
                    "process_time_ms": ms,
                    "num_input_characters": n_chars
                }
                for k,v in out_dict.items():
                    if k == "summary":
                        continue
                    row[k] = v

                if not out_dict["summary"]:
                    logger.warning(f"No summary => {pid}")
                    continue

                txt_dir = os.path.join(output_dir, "text_summaries", subfolder, pid)
                emb_dir = os.path.join(output_dir, "embeddings", subfolder, pid)
                os.makedirs(txt_dir, exist_ok=True)
                os.makedirs(emb_dir, exist_ok=True)

                with open(os.path.join(txt_dir, "path_consult_reports_summary.txt"), 'w', encoding='utf-8') as sf:
                    sf.write(out_dict["summary"])
                emb = self.embeddings.embed_documents([out_dict["summary"]])[0]
                with open(os.path.join(emb_dir, "embedding.pkl"), 'wb') as ef:
                    pickle.dump(emb, ef)

                time_records.append(row)

        if time_records:
            df = pd.DataFrame(time_records)
            csv_path = os.path.join(output_dir, "processing_times.csv")
            if os.path.isfile(csv_path):
                old_df = pd.read_csv(csv_path)
                new_df = pd.concat([old_df, df], ignore_index=True, sort=False)
                new_df = self._cast_numeric(new_df)
                new_df.to_csv(csv_path, index=False)
            else:
                df = self._cast_numeric(df)
                df.to_csv(csv_path, index=False)
            logger.info(f"Saved => {csv_path}")
        else:
            logger.warning("No records to save in processing_times.csv")


    def _cast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_candidates = [
            "process_time_ms", "num_input_characters",
            "text_tokens", "prompt_instr_tokens", "total_needed", "used_ctx",
            "extr_text_tokens","extr_prompt_instr_tokens","extr_total_needed","extr_used_ctx",
            "cot_text_tokens","cot_prompt_instr_tokens","cot_total_needed","cot_used_ctx",
            "path_text_tokens","path_prompt_instr_tokens","path_total_needed","path_used_ctx",
            "cons_text_tokens","cons_prompt_instr_tokens","cons_total_needed","cons_used_ctx",
            "path_extr_text_tokens","path_extr_prompt_instr_tokens","path_extr_total_needed","path_extr_used_ctx",
            "path_cot_text_tokens","path_cot_prompt_instr_tokens","path_cot_total_needed","path_cot_used_ctx",
            "cons_extr_text_tokens","cons_extr_prompt_instr_tokens","cons_extr_total_needed","cons_extr_used_ctx",
            "cons_cot_text_tokens","cons_cot_prompt_instr_tokens","cons_cot_total_needed","cons_cot_used_ctx"
        ]
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df

###############################################################################
# 5. CLI: with new subset arguments (subset_precompute_csv, only_above)
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        "HNC Summarizer Agent7, produce 4 precompute CSV or do normal summarization, plus subset filtering."
    )
    parser.add_argument("--prompts_dir", required=True, help="Directory with JSON prompt files.")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend.")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature.")
    parser.add_argument("--embedding_model", default="ollama", choices=["ollama","openai","google"], help="Emb model.")
    parser.add_argument("--local_model", default="llama3.3:latest", help="Local model name if model_type=local.")
    parser.add_argument("--experiment_mode", type=int, default=1, help="Mode=1..4 => single/two-step + combined/separate.")
    parser.add_argument("--input_dir", required=True, help="Folder with PathologyReports, ConsultRedacted, PathConsCombined subfolders.")
    parser.add_argument("--output_dir", required=True, help="Where to store results.")
    parser.add_argument("--dynamic_ctx_window", action="store_true", help="If set, recalc num_ctx per call.")
    parser.add_argument("--default_ctx", type=int, default=2048, help="Min context if needed tokens < this.")
    parser.add_argument("--margin", type=int, default=500, help="Extra tokens if needed above default_ctx.")
    parser.add_argument("--ollama_context_size", type=int, default=2048, help="Static ctx if dynamic not used.")
    parser.add_argument("--case_ids", default="", help="Comma-separated list of IDs (patient IDs). Omit to do all.")
    parser.add_argument("--single", action="store_true", help="If set, pick one random file/pid from the matched set.")
    parser.add_argument("--precompute_tokens", action="store_true",
                        help="If set, produce 4 CSVs => precompute_tokens_mode1..4.csv, skip summarization.")

    # New subset filtering arguments:
    parser.add_argument("--subset_precompute_csv", default="",
                        help="If provided, e.g. precompute_tokens_mode2.csv. We'll filter rows using --only_above threshold, and run only those file IDs.")
    parser.add_argument("--only_above", type=int, default=0,
                        help="Threshold. We keep rows where extr_total_needed>this or cot_total_needed>this. Then parse file ID.")

    args = parser.parse_args()

    summ = ReportSummarizer(
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

    # If precompute_tokens => do that and exit.
    if args.precompute_tokens:
        os.makedirs(args.output_dir, exist_ok=True)
        summ.precompute_tokens_for_all_modes(args.input_dir, args.output_dir)
        return

    # Otherwise we do normal summarization for the chosen experiment_mode.
    # Possibly subset from a precompute CSV (especially relevant for mode=2, but can be used for others if columns match.)
    final_case_ids = set()

    if args.subset_precompute_csv and os.path.isfile(args.subset_precompute_csv) and args.only_above>0:
        try:
            df_sub = pd.read_csv(args.subset_precompute_csv)
            # we look for extr_total_needed, cot_total_needed columns:
            needed_cols = ["file","extr_total_needed","cot_total_needed"]
            missing = [c for c in needed_cols if c not in df_sub.columns]
            if missing:
                logger.warning(f"subset_precompute_csv missing columns {missing}, ignoring subset filter.")
            else:
                thr = args.only_above
                filtered = df_sub[(df_sub["extr_total_needed"]> thr) | (df_sub["cot_total_needed"]> thr)]
                if not filtered.empty:
                    for fpath in filtered["file"]:
                        baseid = os.path.splitext(fpath)[0]  # e.g. 1115749 if file=1115749.txt
                        final_case_ids.add(baseid)
                    outf = os.path.join(args.output_dir, f"filtered_cases_mode{args.experiment_mode}.csv")
                    filtered.to_csv(outf, index=False)
                    logger.info(f"[Subset] Filtered subset saved => {outf}")
        except Exception as e:
            logger.error(f"Error reading subset_precompute_csv: {e}")

    # Also incorporate --case_ids if user typed them:
    if args.case_ids.strip():
        for cid in args.case_ids.split(","):
            cid = cid.strip()
            if cid:
                final_case_ids.add(cid)

    # If final_case_ids is empty => means "use all"
    summ.process_reports(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        case_ids=list(final_case_ids),
        single=args.single
    )

if __name__ == "__main__":
    import sys
    sys.exit(main())

# Usage example 
# python hnc_reports_agent8.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --local_model "llama3.3:latest" \
#   --embedding_model ollama \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt38" \
#   --experiment_mode 2 \
#   --ollama_context_size 5000 \
#   --subset_precompute_csv "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts/precompute_tokens_mode2.csv" \
#   --only_above 1500
