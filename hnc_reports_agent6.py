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
    raise ValueError("Cannot load 'meta-llama/Llama-3.1-70B' tokenizer. Adjust if needed.")

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
# 3. Merge Summaries (Separate Mode)
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
        dynamic_ctx_window => if True, recalc context for each call
        default_ctx => minimal context if needed < this
        margin => extra tokens if > default_ctx
        ollama_context_size => fallback if not dynamic
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

        # Load prompts
        self.prompt_combined    = self._load_prompt("prompt_path_consult_reports_combined.json")
        self.prompt_extraction  = self._load_prompt("prompt_path_consult_reports_extraction.json")
        self.prompt_cot         = self._load_prompt("prompt_path_consult_reports_cot.json")

        # Init model
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

        # Build runnables
        self.runnable_combined   = self._make_llm_chain(self.prompt_combined)    if self.prompt_combined    else None
        self.runnable_extraction = self._make_llm_chain(self.prompt_extraction)  if self.prompt_extraction  else None
        self.runnable_cot        = self._make_llm_chain(self.prompt_cot)         if self.prompt_cot         else None

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.prompts_dir, filename)
        if not os.path.isfile(path):
            logger.warning(f"No prompt file: {path}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "\n".join(data.get("prompts", []))

    def _init_model_for_tokens(self, needed_tokens: int) -> int:
        """
        If dynamic, re-init model with new ctx
        Return used ctx
        """
        if self.model_type != "local":
            return 0  # non local
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
        """
        Return a function that merges user instructions vs user text separately
        and store text_tokens, prompt_instr_tokens, total_needed, used_ctx
        """
        def chain_func(inputs: Dict[str, str]) -> Dict[str, any]:
            user_text_str = inputs["context"]
            instructions_str = prompt_text  # has {context}, but for counting we measure separately.

            # A) Count user_text tokens
            text_tokens = len(tokenizer.encode(user_text_str, add_special_tokens=False))
            # B) Count instructions w/o user_text
            # We'll remove the {context} placeholder from instructions for token counting
            prompt_noctx = instructions_str.replace("{context}", "")  # just remove the placeholder
            prompt_instr_tokens = len(tokenizer.encode(prompt_noctx, add_special_tokens=False))

            total_needed = text_tokens + prompt_instr_tokens
            used_ctx = self._init_model_for_tokens(total_needed)

            # Now combine them for actual call
            final_prompt = instructions_str.replace("{context}", user_text_str)
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
    # Single-step combined
    ###########################################################################
    def summarize_combined_singlestep(self, text: str) -> Dict[str, any]:
        if not self.runnable_combined:
            logger.warning("No combined prompt loaded.")
            return {"summary": ""}
        result = self.runnable_combined.invoke({"context": text})
        final_summary = enforce_format(result["summary"], ALL_FIELDS)
        result["summary"] = final_summary
        return result

    ###########################################################################
    # Two-step combined
    ###########################################################################
    def summarize_combined_twostep(self, text: str) -> Dict[str, any]:
        out_dict = {
            "summary": ""
        }
        if not (self.runnable_extraction and self.runnable_cot):
            # fallback
            single_res = self.summarize_combined_singlestep(text)
            out_dict["summary"] = single_res["summary"]
            return out_dict

        # step1 => extraction
        ex_res = self.runnable_extraction.invoke({"context": text})
        # step2 => CoT
        cot_res = self.runnable_cot.invoke({"context": text})

        merged = ex_res["summary"] + "\n" + cot_res["summary"]
        final = enforce_format(merged, ALL_FIELDS)
        out_dict["summary"] = final

        # rename keys to extr_* and cot_*
        out_dict["extr_text_tokens"] = ex_res["text_tokens"]
        out_dict["extr_prompt_instr_tokens"] = ex_res["prompt_instr_tokens"]
        out_dict["extr_total_needed"] = ex_res["total_needed"]
        out_dict["extr_used_ctx"] = ex_res["used_ctx"]

        out_dict["cot_text_tokens"] = cot_res["text_tokens"]
        out_dict["cot_prompt_instr_tokens"] = cot_res["prompt_instr_tokens"]
        out_dict["cot_total_needed"] = cot_res["total_needed"]
        out_dict["cot_used_ctx"] = cot_res["used_ctx"]
        return out_dict

    ###########################################################################
    # Single-step separate
    ###########################################################################
    def summarize_separate_singlestep(self, path_text: str, cons_text: str) -> Dict[str, any]:
        pr = self.summarize_combined_singlestep(path_text)
        cr = self.summarize_combined_singlestep(cons_text)
        merged = merge_summaries(pr["summary"], cr["summary"], ALL_FIELDS)

        out_dict = {
            "summary": merged,
            "path_text_tokens": pr["text_tokens"],
            "path_prompt_instr_tokens": pr["prompt_instr_tokens"],
            "path_total_needed": pr["total_needed"],
            "path_used_ctx": pr["used_ctx"],

            "cons_text_tokens": cr["text_tokens"],
            "cons_prompt_instr_tokens": cr["prompt_instr_tokens"],
            "cons_total_needed": cr["total_needed"],
            "cons_used_ctx": cr["used_ctx"]
        }
        return out_dict

    ###########################################################################
    # Two-step separate
    ###########################################################################
    def summarize_separate_twostep(self, path_text: str, cons_text: str) -> Dict[str, any]:
        p_out = self.summarize_combined_twostep(path_text)
        c_out = self.summarize_combined_twostep(cons_text)
        final = merge_summaries(p_out["summary"], c_out["summary"], ALL_FIELDS)

        out_dict = {
            "summary": final,
            # rename path_ => extr/cot
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
            "cons_cot_used_ctx": c_out.get("cot_used_ctx"),
        }
        return out_dict

    ###########################################################################
    # Summarize one patient
    ###########################################################################
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
    # Precompute tokens
    ###########################################################################
    def precompute_tokens(self, input_dir: str, output_dir: str):
        """
        Just gather .txt from PathConsCombined, PathologyReports, ConsultRedacted
        and store file,report_type,num_chars,num_tokens => precompute_tokens.csv
        """
        records = []
        # combined
        cd = os.path.join(input_dir, "PathConsCombined")
        if os.path.isdir(cd):
            for root, _, fs in os.walk(cd):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        rec = {
                            "file": fname,
                            "report_type": "combined",
                            "num_chars": len(txt),
                            "num_tokens": len(tokenizer.encode(txt, add_special_tokens=False))
                        }
                        records.append(rec)
        # pathology
        pd_ = os.path.join(input_dir, "PathologyReports")
        if os.path.isdir(pd_):
            for root, _, fs in os.walk(pd_):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        rec = {
                            "file": fname,
                            "report_type": "pathology",
                            "num_chars": len(txt),
                            "num_tokens": len(tokenizer.encode(txt, add_special_tokens=False))
                        }
                        records.append(rec)
        # consult
        cd_ = os.path.join(input_dir, "ConsultRedacted")
        if os.path.isdir(cd_):
            for root, _, fs in os.walk(cd_):
                for fname in fs:
                    if fname.endswith(".txt"):
                        fp = os.path.join(root, fname)
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        rec = {
                            "file": fname,
                            "report_type": "consult",
                            "num_chars": len(txt),
                            "num_tokens": len(tokenizer.encode(txt, add_special_tokens=False))
                        }
                        records.append(rec)

        if not records:
            logger.warning("No .txt found for precompute.")
            return
        df = pd.DataFrame(records)
        outpath = os.path.join(output_dir, "precompute_tokens.csv")
        df.to_csv(outpath, index=False)
        logger.info(f"Saved => {outpath}")

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
                num_input_toks = len(tokenizer.encode(text, add_special_tokens=False))

                start = time.time()
                out_dict = self.summarize_one_patient(text, "")
                end   = time.time()
                ms = int(round((end - start)*1000))

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

                # save
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

                combined = path_text + "\n\n" + cons_text
                n_chars = len(combined)
                n_toks = len(tokenizer.encode(combined, add_special_tokens=False))

                st = time.time()
                out_dict = self.summarize_one_patient(path_text, cons_text)
                et = time.time()
                ms = int(round((et - st)*1000))

                row = {
                    "file": f"{pid}.txt",
                    "report_type": subfolder,
                    "process_time_ms": ms,
                    "num_input_characters": n_chars,
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
            # If existing, combine columns
            if os.path.isfile(csv_path):
                old_df = pd.read_csv(csv_path)
                new_df = pd.concat([old_df, df], ignore_index=True, sort=False)
                # optional: cast numeric columns to int
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
# 5. CLI
###############################################################################
def main():
    parser = argparse.ArgumentParser("HNC Summarizer Agent6: separate tokens for text vs. prompt_instructions + no patient_id in CSV")
    parser.add_argument("--prompts_dir", required=True, help="Dir with JSON prompts.")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend.")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM temperature.")
    parser.add_argument("--embedding_model", default="ollama", choices=["ollama","openai","google"], help="Emb model.")
    parser.add_argument("--local_model", default="llama3.3:latest", help="Local model name.")
    parser.add_argument("--experiment_mode", type=int, default=1, help="1..4 => single/two-step + combined/separate.")
    parser.add_argument("--input_dir", required=True, help="Parent folder with .txt subfolders.")
    parser.add_argument("--output_dir", required=True, help="Where to store results.")
    parser.add_argument("--dynamic_ctx_window", action="store_true", help="If set, recalc num_ctx per call.")
    parser.add_argument("--default_ctx", type=int, default=2048, help="Minimum ctx if needed tokens < this.")
    parser.add_argument("--margin", type=int, default=500, help="Add margin if needed > default_ctx.")
    parser.add_argument("--ollama_context_size", type=int, default=2048, help="Static ctx if not dynamic.")
    parser.add_argument("--case_ids", default="", help="Comma-separated list of IDs to process.")
    parser.add_argument("--single", action="store_true", help="Pick one random from matched set.")
    parser.add_argument("--precompute_tokens", action="store_true", help="If set, do not summarize, only gather tokens.")
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

    if args.precompute_tokens:
        os.makedirs(args.output_dir, exist_ok=True)
        summ.precompute_tokens(args.input_dir, args.output_dir)
        return

    # gather case_ids
    final_case_ids = set()
    if args.case_ids.strip():
        for cid in args.case_ids.split(","):
            cid = cid.strip()
            if cid:
                final_case_ids.add(cid)

    summ.process_reports(
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
# --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
# --input_dir "/media/yujing/OneTouch3/HNC_Reports" \
# --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Results/Token_Counts" \
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
#   --margin 1000

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
#   --margin 1000

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
#   --margin 1000

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
#   --margin 1000

# NOW DO THE PRECOMPUTATION OF THE TOKENS FOR EXPERIMENT 2 (COMBINED + TWO-STEP SUMMARIES)
# GET A LIST OF CASE IDS WHOSE TOTAL TOKENS NEEDED WERE BELOW VERSUS ABOVE 2048 THE CHATOLLAMA DEFAULT 
# THEN MAKE SURE THE CASE_ID ARGUMENT ACN PROCESS THIS .CSV LIST OF THEM ONLY IN EXP14. 

# FOR THE COLLECTION OF FINAL RESULTS THEY CAN BE MERGED FROM EXP13 FOR THE ONES BELOW 2048 AND THE ONES ABOVE 2048 FROM EXP14.
# PERHAPS SAVE THIS COMBINED FINAL UNIQUE 882 CASES INTO ANOTHER FOLDER? FOR PAPER 2 PROCESSING 

# REDO THE 50 CASE SELECTIONS FROM AN ACTUAL RANGE OF THE INPUT TOKENS BASED ON EXP MODE2 INSTEAD OF THE NUMBER OF NOT INFERRED! 
# DISCUSS WITH LAYA AND FARHOOD AND BE DONE!