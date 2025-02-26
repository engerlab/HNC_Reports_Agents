#!/usr/bin/env python3
"""
Summarize HNC Reports with Experimental Modes for Prompt Engineering

This script processes reports from:
  - PathologyReports/ (for pathology_reports)
  - ConsultRedacted/ (for consultation_notes)
  - PathConsCombined/ (for treatment_plan_outcomepred, path_consult_reports, and cot_treatment_plan_outcomepred)

It supports two experimental modes for path_consult_reports:
  1. Combined mode: A single combined prompt (fields that do not require additional reasoning are handled with a “Not inferred” default).
  2. Separated mode: Two separate prompts (one for fields that do not require chain-of-thought and one for those that require reasoning) whose outputs are then merged.

Additional features:
  --single: Process only one random file per subfolder.
  --case_id: Process a specific case (file name without extension).
  --prompt_mode: Optional suffix for prompt JSON files (e.g., "combined" or "separated").

The processing_times.csv file will record only the time taken for the summarization (in milliseconds), as well as the number of input characters and tokens.
  
Usage examples:
  Full processing (all cases) with combined prompt mode:
    python hnc_reports_agent5.py \
      --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
      --model_type local \
      --temperature 0.8 \
      --input_dir "/media/yujing/One Touch3/HNC_Reports" \
      --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpFull1" \
      --embedding_model ollama \
      --report_type "path_consult_reports" \
      --local_model "llama3.3:latest" \
      --prompt_mode "combined"

  Single-case experiment (random case):
    python hnc_reports_agent5.py \
      --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
      --model_type local \
      --temperature 0.8 \
      --input_dir "/media/yujing/One Touch3/HNC_Reports" \
      --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPrompt1" \
      --embedding_model ollama \
      --report_type "path_consult_reports" \
      --local_model "llama3.3:latest" \
      --prompt_mode "separated" \
      --single

  Single-case experiment (specific case):
    python hnc_reports_agent5.py \
      --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
      --model_type local \
      --temperature 0.8 \
      --input_dir "/media/yujing/One Touch3/HNC_Reports" \
      --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPrompt2" \
      --embedding_model ollama \
      --report_type "path_consult_reports" \
      --local_model "llama3.3:latest" \
      --prompt_mode "combined" \
      --case_id "1115749"

Note: The script uses a Hugging Face tokenizer (bert-base-uncased) to count tokens.
"""

import os
import json
import re
import argparse
import logging
import pickle
from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import random
import signal

# Import Hugging Face tokenizer to count tokens.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# LangChain
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

# Model-specific imports (adapt as needed)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

##############################################################################
# 1. Field Definitions (for the combined extraction)
##############################################################################
# Note: The following fields are used in the combined prompt.
PATH_CONS_FIELDS = [
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

##############################################################################
# 2. Regex Patterns
##############################################################################
PATTERNS = {
    "Sex": r"^Sex:\s*(Male|Female|Other|Not inferred)\s*$",
    "Anatomic_Site_of_Lesion": r"^Anatomic_Site_of_Lesion:\s*(.*)$",
    "Pathological_TNM": r"^Pathological_TNM:\s*(.*)$",
    "Clinical_TNM": r"^Clinical_TNM:\s*(.*)$",
    "Primary_Tumor_Size": r"^Primary_Tumor_Size:\s*(.*)$",
    "Tumor_Type_Differentiation": r"^Tumor_Type_Differentiation:\s*(.*)$",
    "Pathology_Details": r"^Pathology_Details:\s*(.*)$",
    "Lymph_Node_Status_Presence_Absence": r"^Lymph_Node_Status_Presence_Absence:\s*(Presence|Absence|Suspected|Not inferred)\s*$",
    "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes": r"^Lymph_Node_Status_Number_of_Positve_Lymph_Nodes:\s*(\d+|Not inferred)\s*$",
    "Lymph_Node_Status_Extranodal_Extension": r"^Lymph_Node_Status_Extranodal_Extension:\s*(Yes|No|Not inferred)\s*$",
    "Resection_Margins": r"^Resection_Margins:\s*(Positive|Negative|Not inferred)\s*$",
    "p16_Status": r"^p16_Status:\s*(Positive|Negative|Not inferred)\s*$",
    "Immunohistochemical_profile": r"^Immunohistochemical_profile:\s*(.*)$",
    "EBER_Status": r"^EBER_Status:\s*(Positive|Negative|Not inferred)\s*$",
    "Lymphovascular_Invasion_Status": r"^Lymphovascular_Invasion_Status:\s*(Present|Absent|Not inferred)\s*$",
    "Perineural_Invasion_Status": r"^Perineural_Invasion_Status:\s*(Present|Absent|Not inferred)\s*$",
    "Smoking_History": r"^Smoking_History:\s*(.*)$",
    "Alcohol_Consumption": r"^Alcohol_Consumption:\s*(.*)$",
    "Pack_Years": r"^Pack_Years:\s*(\d+|Not inferred)\s*$",
    "Patient_Symptoms_at_Presentation": r"^Patient_Symptoms_at_Presentation:\s*(.*)$",
    "Recommendations": r"^Recommendations:\s*(.*)$",
    "Follow_Up_Plans": r"^Follow_Up_Plans:\s*(.*)$",
    "HPV_Status": r"^HPV_Status:\s*(Positive|Negative|Not inferred)\s*$",
    "Patient_History_Status_Prior_Conditions": r"^Patient_History_Status_Prior_Conditions:\s*(.*)$",
    "Patient_History_Status_Previous_Treatments": r"^Patient_History_Status_Previous_Treatments:\s*(.*)$",
    "Clinical_Assessments_Radiological_Lesions": r"^Clinical_Assessments_Radiological_Lesions:\s*(.*)$",
    "Clinical_Assessments_SUV_from_PET_scans": r"^Clinical_Assessments_SUV_from_PET_scans:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "Charlson_Comorbidity_Score": r"^Charlson_Comorbidity_Score:\s*(\d+|Not inferred)\s*$",
    "Karnofsky_Performance_Status": r"^Karnofsky_Performance_Status:\s*(\d+|Not inferred)\s*$",
    "ECOG_Performance_Status": r"^ECOG_Performance_Status:\s*(\d+|Not inferred)\s*$"
}

##############################################################################
# 3. Helper Function to Enforce Format
##############################################################################
def enforce_format(summary: str, expected_fields: List[str]) -> str:
    """
    Ensure the output has exactly one line per expected field (FieldName: Value).
    For any field not present, force 'Not inferred'.
    """
    lines = summary.splitlines()
    field_lines = {}
    for field in expected_fields:
        pattern = f"^{field}:"  # must start with the field name followed by a colon
        found_line = None
        for line in lines:
            if re.match(pattern, line.strip()):
                found_line = line.strip()
                break
        field_lines[field] = found_line if found_line else f"{field}: Not inferred"
    return "\n".join([field_lines[field] for field in expected_fields])

##############################################################################
# 4. Summarizer Class
##############################################################################
class ReportSummarizer:
    def __init__(
        self,
        prompts_dir: str,
        model_type: str = "local",
        temperature: float = 0.8,
        embedding_model: str = "ollama",
        local_model: str = "llama3.3:latest",
        prompt_mode: str = ""
    ):
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.embedding_model = embedding_model.lower()
        self.local_model = local_model
        self.prompt_mode = prompt_mode.lower()  # e.g., "combined" or "separated"

        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Invalid prompts_dir: {prompts_dir}")
        self.prompts_dir = prompts_dir

        # Load prompts for known report types and sub-prompts for path_consult_reports
        known_rtypes = [
            "pathology_reports",
            "consultation_notes",
            "treatment_plan_outcomepred",
            "path_consult_reports"
        ]
        # For the two-step approach for path_consult_reports, we load two separate prompt texts.
        self.prompts = {}
        for rtype in known_rtypes:
            self.prompts[rtype] = self.load_prompt(rtype)
        # For two-step path_consult_reports extraction (separated prompts)
        self.prompts["path_consult_reports_extraction"] = self.load_prompt("path_consult_reports_extraction")
        self.prompts["path_consult_reports_cot"] = self.load_prompt("path_consult_reports_cot")

        # Initialize LLM model
        if self.model_type == "local":
            self.model = ChatOllama(model=self.local_model, temperature=self.temperature)
        elif self.model_type == "gpt":
            self.model = ChatOpenAI(model="gpt-4", temperature=self.temperature)
        elif self.model_type == "gemini":
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Initialize embeddings
        if self.embedding_model == "ollama":
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        elif self.embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif self.embedding_model == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        else:
            logger.warning(f"Unknown embedding_model: {self.embedding_model}. Defaulting to Ollama.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build chain map for single-step prompts (for report types other than path_consult_reports)
        self.chain_map = {}
        for rtype in known_rtypes:
            if self.prompts[rtype]:
                self.chain_map[rtype] = self.make_llm_runnable(self.prompts[rtype])
    
    def load_prompt(self, rtype: str) -> str:
        """
        Load prompt text from JSON file. If prompt_mode is set, try with that suffix.
        """
        base_filename = f"prompt_{rtype}"
        candidates = []
        if self.prompt_mode:
            candidates.append(os.path.join(self.prompts_dir, f"{base_filename}_{self.prompt_mode}.json"))
        candidates.append(os.path.join(self.prompts_dir, f"{base_filename}.json"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                with open(candidate, 'r') as f:
                    data = json.load(f)
                prompts_list = data.get("prompts", [])
                logger.info(f"Loaded prompt for {rtype} from {candidate}")
                return "\n".join(prompts_list)
        logger.warning(f"No prompt file found for {rtype} (candidates: {candidates})")
        return ""
    
    def make_llm_runnable(self, prompt: str) -> RunnableLambda:
        def llm_runnable(inputs: Dict[str, str]) -> Dict[str, str]:
            context = inputs["context"]
            logger.debug(f"Summarizing text of length {len(context)} for {inputs.get('report_type', 'unknown')}")
            try:
                final_prompt = prompt.replace("{context}", context)
                result = self.model.invoke([HumanMessage(content=final_prompt)]).content
                return {"summary": result}
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return {"summary": ""}
        return RunnableLambda(llm_runnable)
    
    def run_llm_prompt(self, rtype: str, text: str) -> str:
        if rtype not in self.chain_map:
            logger.warning(f"No prompt found for {rtype}")
            return ""
        runnable = self.chain_map[rtype]
        resp = runnable.invoke({"context": text, "report_type": rtype})
        summary = resp.get("summary", "").strip()
        logger.debug(f"[{rtype}] Summarize output (first 80 chars): {summary[:80]}...")
        return summary
    
    def summarize_path_consult_two_step(self, report_text: str) -> str:
        extraction_summary = self.run_llm_prompt("path_consult_reports_extraction", report_text)
        cot_summary = self.run_llm_prompt("path_consult_reports_cot", report_text)
        combined_raw = extraction_summary + "\n" + cot_summary
        final_text = enforce_format(combined_raw, PATH_CONS_FIELDS)
        return final_text

    def summarize_report(self, report_text: str, report_type: str) -> Optional[str]:
        if report_type == "path_consult_reports":
            if (self.prompts.get("path_consult_reports_extraction") and self.prompts.get("path_consult_reports_cot")):
                summary = self.summarize_path_consult_two_step(report_text)
            else:
                logger.warning("[path_consult_reports] Two-step prompts not found, using single-step fallback.")
                summary = self.run_llm_prompt("path_consult_reports", report_text)
                summary = enforce_format(summary, PATH_CONS_FIELDS)
        else:
            summary = self.run_llm_prompt(report_type, report_text)
            if report_type in ["pathology_reports", "consultation_notes"]:
                if report_type == "pathology_reports":
                    summary = enforce_format(summary, PATH_CONS_FIELDS[:len(PATH_CONS_FIELDS)//2])  # For example purposes.
                elif report_type == "consultation_notes":
                    summary = enforce_format(summary, PATH_CONS_FIELDS[len(PATH_CONS_FIELDS)//2:])
        return summary if summary else None

    def process_reports(self, input_dir: str, output_dir: str, report_types: List[str],
                        single: bool = False, case_id: Optional[str] = None):
        os.makedirs(output_dir, exist_ok=True)
        time_data = []
        def process_folder(folder_path: str) -> List[str]:
            files = []
            if os.path.isdir(folder_path):
                files = [os.path.join(root, fname) for root, _, fs in os.walk(folder_path)
                         for fname in fs if fname.endswith(".txt")]
                if case_id:
                    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] == case_id]
                    if not files:
                        logger.warning(f"No file found with case ID {case_id} in {folder_path}")
                elif single and files:
                    files = [random.choice(files)]
            return files

        folder_map = {
            "pathology_reports": os.path.join(input_dir, "PathologyReports"),
            "consultation_notes": os.path.join(input_dir, "ConsultRedacted"),
            "treatment_plan_outcomepred": os.path.join(input_dir, "PathConsCombined"),
            "path_consult_reports": os.path.join(input_dir, "PathConsCombined"),
            "cot_treatment_plan_outcomepred": os.path.join(input_dir, "PathConsCombined")
        }

        for rtype in report_types:
            folder = folder_map.get(rtype)
            if not folder or not os.path.isdir(folder):
                logger.warning(f"No valid folder for {rtype} in {input_dir}. Skipping.")
                continue
            file_list = process_folder(folder)
            for path in file_list:
                fname = os.path.basename(path)
                logger.info(f"Processing file: {fname} for report type: {rtype}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    # Record input characteristics
                    num_chars = len(text)
                    num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    # Measure summarization time only
                    start_time = time.time()
                    summary = self.summarize_report(text, rtype)
                    end_time = time.time()
                    elapsed_ms = int(round((end_time - start_time) * 1000))
                    time_data.append({
                        "file": fname,
                        "report_type": rtype,
                        "process_time_ms": elapsed_ms,
                        "num_input_characters": num_chars,
                        "num_input_tokens": num_tokens
                    })
                    if not summary:
                        logger.warning(f"No summary produced for {fname}.")
                        continue

                    # Prepare output directories (only saving text_summaries and embeddings)
                    patient_id = os.path.splitext(fname)[0]
                    subdirs = {
                        "text_summaries": os.path.join(output_dir, "text_summaries", rtype, patient_id),
                        "embeddings": os.path.join(output_dir, "embeddings", rtype, patient_id)
                    }
                    for sd in subdirs.values():
                        os.makedirs(sd, exist_ok=True)
                    with open(os.path.join(subdirs["text_summaries"], f"{rtype}_summary.txt"), 'w', encoding='utf-8') as sf:
                        sf.write(summary)
                    emb = self.embeddings.embed_documents([summary])[0]
                    with open(os.path.join(subdirs["embeddings"], f"{rtype}_embedding.pkl"), 'wb') as ef:
                        pickle.dump(emb, ef)
                except Exception as e:
                    logger.error(f"Error processing {fname}: {e}")

        try:
            if time_data:
                pd.DataFrame(time_data).to_csv(os.path.join(output_dir, "processing_times.csv"), index=False)
                logger.info(f"Processing times saved to {os.path.join(output_dir, 'processing_times.csv')}")
        except Exception as e:
            logger.error(f"Failed to save processing_times.csv: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize HNC Reports with Experimental Modes for Combined Extraction.\n"
                    "Options:\n"
                    "  --report_type: Comma-separated list of report types (e.g., 'path_consult_reports').\n"
                    "  --prompt_mode: Optional suffix for prompt JSON files (e.g., 'combined' or 'separated').\n"
                    "  --single: If set, process one random file per folder.\n"
                    "  --case_id: (Optional) Process a specific case by filename (without extension)."
    )
    parser.add_argument("--prompts_dir", required=True, help="Directory containing prompt JSON files.")
    parser.add_argument("--model_type", default="local", choices=["local", "gpt", "gemini"], help="LLM backend.")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature.")
    parser.add_argument("--input_dir", required=True, help="Parent directory with subfolders.")
    parser.add_argument("--output_dir", required=True, help="Output directory for results.")
    parser.add_argument("--embedding_model", type=str, default="ollama", choices=["ollama", "openai", "google"], help="Embedding model.")
    parser.add_argument("--report_type", type=str, default="all", help="Comma-separated list of report types.")
    parser.add_argument("--local_model", type=str, default="llama3.3:latest", help="Local model name if model_type='local'.")
    parser.add_argument("--single", action="store_true", help="Process one random file per folder.")
    parser.add_argument("--case_id", type=str, default="", help="Specify a case ID (filename without extension) to process.")
    parser.add_argument("--prompt_mode", type=str, default="", help="Optional suffix for prompt JSON files (e.g., 'combined').")
    args = parser.parse_args()

    if args.report_type.lower() == "all":
        report_types = [
            "pathology_reports",
            "consultation_notes",
            "treatment_plan_outcomepred",
            "path_consult_reports",
            "cot_treatment_plan_outcomepred"
        ]
    else:
        report_types = [rt.strip().lower() for rt in args.report_type.split(",")]

    logger.info(f"Selected report types: {report_types}")
    os.makedirs(args.output_dir, exist_ok=True)
    summarizer = ReportSummarizer(
        prompts_dir=args.prompts_dir,
        model_type=args.model_type,
        temperature=args.temperature,
        embedding_model=args.embedding_model,
        local_model=args.local_model,
        prompt_mode=args.prompt_mode
    )
    case_id = args.case_id.strip() if args.case_id.strip() else None
    summarizer.process_reports(args.input_dir, args.output_dir, report_types, single=args.single, case_id=case_id)

if __name__ == "__main__":
    os.setpgrp()  # Ensure all child processes are in the same group for termination.
    main()
