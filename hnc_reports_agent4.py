#!/usr/bin/env python3
"""
Summarize HNC Reports with separate sub-prompts for path_consult_reports:
  - Splits the summarization into two prompts (non-CoT and CoT).
  - Combines the results into a single summary with all required fields.

Usage:
  --report_type: Comma-separated list of report types to process. Options:
      pathology_reports,
      consultation_notes,
      treatment_plan_outcomepred,
      path_consult_reports,
      cot_treatment_plan_outcomepred.
      Default is "all".
  --local_model: If using local model, specify the model name (default: "llama3.3:latest").
  --prompt_mode: (Optional) If you want to use a specific suffix on your prompt files, set it here.
  --single: (Optional) Process one random file per folder.
  --case_id: (Optional) Process a specific case by filename without extension.
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

# LangChain
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

# Model-specific (example imports; adapt to your local environment)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

##############################################################################
# 1. Field Definitions
##############################################################################
PATHOLOGY_FIELDS = [
    "Age",
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

]

CONSULTATION_FIELDS = [
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

# Combined fields: merge pathology and consultation fields (removing duplicates)
PATH_CONS_FIELDS = list(dict.fromkeys(PATHOLOGY_FIELDS + CONSULTATION_FIELDS))

##############################################################################
# 2. Regex Patterns
##############################################################################
PATTERNS = {
    "Age": r"^Age:\s*(\d+|Not inferred)\s*$",
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
    "Treatment_Recommendations": r"^Treatment_Recommendations:\s*(.*)$",
    "Follow_Up_Plans": r"^Follow_Up_Plans:\s*(.*)$",
    "HPV_Status": r"^HPV_Status:\s*(Positive|Negative|Not inferred)\s*$",
    "Patient_History_Status_Prior_Conditions": r"^Patient_History_Status_Prior_Conditions:\s*(.*)$",
    "Patient_History_Status_Previous_Treatments": r"^Patient_History_Status_Previous_Treatments:\s*(.*)$",
    "Clinical_Assessments_Radiological_Lesions": r"^Clinical_Assessments_Radiological_Lesions:\s*(.*)$",
    "Clinical_Assessments_SUV_from_PET_scans": r"^Clinical_Assessments_SUV_from_PET_scans:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "Charlson_Comorbidity_Score": r"^Charlson_Comorbidity_Score:\s*(\d+|Not inferred)\s*$",
    "Karnofsky_Performance_Status": r"^Karnofsky_Performance_Status:\s*(100|90|80|70|60|50|40|30|20|10|0|Not inferred)\s*$",
    "ECOG_Performance_Status": r"^ECOG_Performance_Status:\s*(0|1|2|3|4|5|Not inferred)\s*$"
}

##############################################################################
# 3. Extraction & Validation Functions
##############################################################################
def extract_tabular_data(summary: str, report_type: str) -> Dict[str, Any]:
    """Use regex to capture each field from the summary text."""
    if report_type == "pathology_reports":
        fields = PATHOLOGY_FIELDS
    elif report_type == "consultation_notes":
        fields = CONSULTATION_FIELDS
    elif report_type == "path_consult_reports":
        fields = PATH_CONS_FIELDS
    else:
        logger.warning(f"Extraction not applicable for report type: {report_type}")
        return {}
    extracted = {}
    for field in fields:
        pattern = PATTERNS.get(field)
        if pattern:
            match = re.search(pattern, summary, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted[field] = match.group(1).strip()
                logger.debug(f"[{report_type}] Matched '{field}': {extracted[field][:80]}...")
            else:
                logger.debug(f"[{report_type}] No match for '{field}'. Setting 'Not inferred'")
                extracted[field] = "Not inferred"
        else:
            logger.debug(f"[{report_type}] No pattern for '{field}'. Setting 'Not inferred'")
            extracted[field] = "Not inferred"
    return extracted

def validate_extracted_data(data: Dict[str, Any], report_type: str) -> bool:
    if report_type in ["treatment_plan_outcomepred", "cot_treatment_plan_outcomepred"]:
        return True
    if report_type == "pathology_reports":
        required_fields = PATHOLOGY_FIELDS
    elif report_type == "consultation_notes":
        required_fields = CONSULTATION_FIELDS
    elif report_type == "path_consult_reports":
        required_fields = PATH_CONS_FIELDS
    else:
        logger.debug(f"[{report_type}] Not recognized in validation.")
        return False
    for field in required_fields:
        if field not in data:
            logger.debug(f"[{report_type}] Missing field '{field}' in data.")
            return False
    return True

def normalize_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    df = df.replace("Not inferred", np.nan)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df

def encode_structured_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    """OneHot + numeric encoding. (Not always used in these experiments, but here for reference.)"""
    logger.debug(f"[{report_type}] Starting encode with shape: {df.shape}")
    df_encoded = df.copy()
    if report_type == "pathology_reports":
        cat_cols = [
            "Sex", "Anatomic_Site_of_Lesion", "Pathological_TNM",
            "Clinical_TNM", "Pathology_Details", "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Extranodal_Extension", "Resection_Margins", "p16_Status",
            "Immunohistochemical_profile", "EBER_Status", "Lymphovascular_Invasion_Status",
            "Perineural_Invasion_Status"
        ]
        num_cols = ["Age", "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes"]
    elif report_type == "consultation_notes":
        cat_cols = [
            "Smoking_History", "Alcohol_Consumption", "Patient_Symptoms_at_Presentation",
            "Treatment_Recommendations", "Follow_Up_Plans", "HPV_Status", "Patient_History_Status_Prior_Conditions",
            "Patient_History_Status_Previous_Treatments", "Clinical_Assessments_Radiological_Lesions"
        ]
        num_cols = ["Pack_Years", "Clinical_Assessments_SUV_from_PET_scans", "Charlson_Comorbidity_Score"]
        num_cols += ["Karnofsky_Performance_Status", "ECOG_Performance_Status"]
    elif report_type == "path_consult_reports":
        cat_cols = [
            "Sex", "Anatomic_Site_of_Lesion", "Pathological_TNM",
            "Clinical_TNM", "Pathology_Details", "Tumor_Type_Differentiation", "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Extranodal_Extension", "Resection_Margins", "p16_Status",
            "Immunohistochemical_profile", "EBER_Status", "Lymphovascular_Invasion_Status",
            "Perineural_Invasion_Status",
            "Smoking_History", "Alcohol_Consumption", "Patient_Symptoms_at_Presentation",
            "Treatment_Recommendations", "Follow_Up_Plans", "HPV_Status", "Patient_History_Status_Prior_Conditions",
            "Patient_History_Status_Previous_Treatments", "Clinical_Assessments_Radiological_Lesions"
        ]
        # NOTE: Some fields appear numeric but might be strings. Adjust if needed.
        num_cols = [
            "Age",
            "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes",
            "Pack_Years",
            "Clinical_Assessments_SUV_from_PET_scans",
            "Charlson_Comorbidity_Score",
            "Karnofsky_Performance_Status",
            "ECOG_Performance_Status"
        ]
    else:
        logger.debug(f"[{report_type}] Unknown type for encoding. Returning unmodified.")
        return df_encoded

    # Convert numerics
    for col in num_cols:
        if col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            logger.debug(f"[{report_type}] Numeric conversion '{col}': {df_encoded[col].head(2).tolist()}")
    if num_cols:
        imp = SimpleImputer(strategy='median')
        df_encoded[num_cols] = imp.fit_transform(df_encoded[num_cols])

    # One-hot encode categoricals
    cat_cols_existing = [c for c in cat_cols if c in df_encoded.columns]
    if cat_cols_existing:
        enc = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        arr = enc.fit_transform(df_encoded[cat_cols_existing])
        encoded_df = pd.DataFrame(arr, columns=enc.get_feature_names_out(cat_cols_existing))
        df_encoded.drop(columns=cat_cols_existing, inplace=True)
        df_encoded.reset_index(drop=True, inplace=True)
        encoded_df.reset_index(drop=True, inplace=True)
        if len(df_encoded) != len(encoded_df):
            logger.debug(f"[{report_type}] Mismatch in rows after encoding: {len(df_encoded)} vs {len(encoded_df)}")
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    logger.debug(f"[{report_type}] Final shape after encoding: {df_encoded.shape}")
    return df_encoded

##############################################################################
# 4. Helper Function to Enforce Format
##############################################################################
def enforce_format(summary: str, expected_fields: List[str]) -> str:
    """
    Ensures the output has exactly one line per expected field (FieldName: Value).
    Any field not found is forced to 'Not inferred'.
    """
    lines = summary.splitlines()
    field_lines = {}
    for field in expected_fields:
        found_line = None
        pattern = f"^{field}:"
        for line in lines:
            if re.match(pattern, line.strip()):
                found_line = line.strip()
                break
        if found_line:
            field_lines[field] = found_line
        else:
            field_lines[field] = f"{field}: Not inferred"
    return "\n".join([field_lines[field] for field in expected_fields])

##############################################################################
# 5. Summarizer Class (with two-step logic for path_consult_reports)
##############################################################################
class ReportSummarizer:
    def __init__(
        self,
        prompts_dir: str,
        model_type: str = "local",
        temperature: float = 0.3,
        embedding_model: str = "ollama",
        local_model: str = "llama3.3:latest",
        prompt_mode: str = ""
    ):
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.embedding_model = embedding_model.lower()
        self.local_model = local_model
        self.prompt_mode = prompt_mode.lower()  # e.g., "combined" or empty for default

        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Invalid prompts_dir: {prompts_dir}")

        self.prompts_dir = prompts_dir

        # Load prompts for each known report type and variations
        #   e.g. prompt_path_consult_reports_extraction.json, prompt_path_consult_reports_cot.json
        #   plus others from the older approach (prompt_path_consult_reports.json, etc.)
        self.prompts = {}
        # We'll expect these filenames:
        #  "prompt_{rtype}.json", "prompt_{rtype}_{prompt_mode}.json"
        #  For the 2-step path_consult, we do "prompt_path_consult_reports_extraction.json"
        #    + optional mode suffix, and "prompt_path_consult_reports_cot.json" + optional mode suffix.
        known_rtypes = [
            "pathology_reports",
            "consultation_notes",
            "treatment_plan_outcomepred",
            "path_consult_reports",
            "cot_treatment_plan_outcomepred",
            # Additional sub-split for path_consult_reports
            "path_consult_reports_extraction",
            "path_consult_reports_cot"
        ]
        for rtype in known_rtypes:
            self.prompts[rtype] = self.load_prompt(rtype)

        # Initialize the chosen model
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
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
        else:
            logger.warning(f"Unknown embedding_model: {self.embedding_model}. Defaulting to Ollama.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build a map of rtype -> prompt-based function (single-step).
        # For path_consult_reports, we will do something special below (two-step).
        self.chain_map = {}
        for rtype, prompt_text in self.prompts.items():
            if prompt_text:
                self.chain_map[rtype] = self.make_llm_runnable(prompt_text)

    def load_prompt(self, rtype: str) -> str:
        """
        Attempt to load the prompt text for a given rtype from JSON in self.prompts_dir.
        We check:
           prompt_{rtype}_{prompt_mode}.json
        then fallback to:
           prompt_{rtype}.json
        Returns the combined prompt as a single string or empty if not found.
        """
        # e.g. prompt_path_consult_reports_extraction_combined.json or prompt_path_consult_reports_extraction.json
        base_filename = f"prompt_{rtype}"
        candidates = []
        if self.prompt_mode:
            candidates.append(os.path.join(self.prompts_dir, f"{base_filename}_{self.prompt_mode}.json"))
        candidates.append(os.path.join(self.prompts_dir, f"{base_filename}.json"))

        for cfile in candidates:
            if os.path.isfile(cfile):
                with open(cfile, 'r') as f:
                    data = json.load(f)
                prompts_list = data.get("prompts", [])
                logger.info(f"Loaded prompt for {rtype} from {cfile}")
                # Join them into one text
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
        """
        Generic method to run a single-step LLM summarization for rtype and text.
        Returns the raw summary string.
        """
        if rtype not in self.chain_map:
            logger.warning(f"No prompt found for {rtype}")
            return ""
        runnable = self.chain_map[rtype]
        resp = runnable.invoke({"context": text, "report_type": rtype})
        summary = resp.get("summary", "").strip()
        logger.debug(f"[{rtype}] Summarize output (first 80 chars): {summary[:80]}...")
        return summary

    def summarize_path_consult_two_step(self, report_text: str) -> str:
        """
        Two-step approach for path_consult_reports:
          1) Summarize with the 'path_consult_reports_extraction' prompt (fields that do NOT need chain-of-thought).
          2) Summarize with the 'path_consult_reports_cot' prompt (fields requiring CoT reasoning).
          3) Merge results into one summary (all fields).
        """
        # Step 1: Non-CoT extraction
        extraction_summary = self.run_llm_prompt("path_consult_reports_extraction", report_text)
        # Step 2: CoT extraction
        cot_summary = self.run_llm_prompt("path_consult_reports_cot", report_text)

        # Combine line-by-line. Easiest is to just put them together and then re-run enforce_format.
        combined_raw = extraction_summary + "\n" + cot_summary
        # Now enforce the final path_consult_reports field ordering:
        final_text = enforce_format(combined_raw, PATH_CONS_FIELDS)
        return final_text

    def summarize_report(self, report_text: str, report_type: str) -> Optional[str]:
        """
        Called for each file. If it's path_consult_reports, do the two-step approach.
        Otherwise, do the single-step approach using self.chain_map.
        Then enforce final format if it's in the set of known structured fields.
        """
        if report_type == "path_consult_reports":
            # Use the 2-step approach if the sub-prompts exist
            # If we don't have them, fallback to single-step
            if (self.prompts["path_consult_reports_extraction"] and
                    self.prompts["path_consult_reports_cot"]):
                summary = self.summarize_path_consult_two_step(report_text)
            else:
                logger.warning("[path_consult_reports] Two sub-prompts not found, using single-step fallback.")
                summary = self.run_llm_prompt("path_consult_reports", report_text)
                summary = enforce_format(summary, PATH_CONS_FIELDS)
        else:
            # Normal single-step
            summary = self.run_llm_prompt(report_type, report_text)
            if report_type in ["pathology_reports", "consultation_notes", "path_consult_reports"]:
                if report_type == "pathology_reports":
                    summary = enforce_format(summary, PATHOLOGY_FIELDS)
                elif report_type == "consultation_notes":
                    summary = enforce_format(summary, CONSULTATION_FIELDS)
                else:
                    summary = enforce_format(summary, PATH_CONS_FIELDS)

        return summary if summary else None

    def process_reports(self, input_dir: str, output_dir: str, report_types: List[str],
                        single: bool = False, case_id: Optional[str] = None):
        """
        - 'pathology_reports' => read from PathologyReports/ in input_dir
        - 'consultation_notes' => read from ConsultRedacted/ in input_dir
        - 'treatment_plan_outcomepred', 'path_consult_reports', 'cot_treatment_plan_outcomepred'
          => read from PathConsCombined/ in input_dir.
        If single is True, process only one file per folder.
        If case_id is provided, process only that file.
        """
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
                start_time = time.time()
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    summary = self.summarize_report(text, rtype)
                    if not summary:
                        logger.warning(f"No summary produced for {fname}.")
                        continue

                    # Timing stats
                    num_chars = len(text)
                    num_tokens = len(text.split())
                    end_time = time.time()
                    elapsed_ms = int(round((end_time - start_time) * 1000))
                    time_data.append({
                        "file": fname,
                        "report_type": rtype,
                        "process_time_ms": elapsed_ms,
                        "num_input_characters": num_chars,
                        "num_input_tokens": num_tokens
                    })

                    # Prepare output directories
                    patient_id = os.path.splitext(fname)[0]
                    subdirs = {
                        "text_summaries": os.path.join(output_dir, "text_summaries", rtype, patient_id),
                        "embeddings": os.path.join(output_dir, "embeddings", rtype, patient_id),
                        "structured_data": os.path.join(output_dir, "structured_data", rtype, patient_id),
                        "structured_data_encoded": os.path.join(output_dir, "structured_data_encoded", rtype, patient_id)
                    }
                    for sd in subdirs.values():
                        os.makedirs(sd, exist_ok=True)

                    # 1) Save final text summary
                    with open(os.path.join(subdirs["text_summaries"], f"{rtype}_summary.txt"), 'w', encoding='utf-8') as sf:
                        sf.write(summary)

                    # 2) Save embeddings
                    emb = self.embeddings.embed_documents([summary])[0]
                    with open(os.path.join(subdirs["embeddings"], f"{rtype}_embedding.pkl"), 'wb') as ef:
                        pickle.dump(emb, ef)

                    # 3) (Optional) Extract structured data CSV
                    #    You could store or skip if you only want text_summaries + embeddings.
                    extracted_data = extract_tabular_data(summary, rtype)
                    df = pd.DataFrame([extracted_data])
                    df_normalized = normalize_data(df, rtype)
                    df_normalized.to_csv(os.path.join(subdirs["structured_data"], f"{rtype}_structured.csv"), index=False)

                    # 4) (Optional) Encoded CSV
                    df_encoded = encode_structured_data(df_normalized, rtype)
                    df_encoded.to_csv(os.path.join(subdirs["structured_data_encoded"], f"{rtype}_structured_encoded.csv"), index=False)

                except Exception as e:
                    logger.error(f"Error processing {fname}: {e}")

        # Save timing data
        try:
            if time_data:
                pd.DataFrame(time_data).to_csv(os.path.join(output_dir, "processing_times.csv"), index=False)
                logger.info(f"Processing times saved to {output_dir}/processing_times.csv")
        except Exception as e:
            logger.error(f"Failed to save processing_times.csv: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize HNC Reports. Splits path_consult_reports into two sub-prompts (non-CoT & CoT)."
    )
    parser.add_argument("--prompts_dir", required=True, help="Directory containing prompt JSON files.")
    parser.add_argument("--model_type", default="local", choices=["local", "gpt", "gemini"], help="LLM backend.")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature.")
    parser.add_argument("--input_dir", required=True, help="Parent directory with subfolders.")
    parser.add_argument("--output_dir", required=True, help="Output directory for results.")
    parser.add_argument("--embedding_model", type=str, default="ollama", choices=["ollama", "openai", "google"], help="Embedding model.")
    parser.add_argument("--report_type", type=str, default="all", help="Comma-separated list of report types.")
    parser.add_argument("--local_model", type=str, default="llama3.3:latest", help="Local model name if model_type='local'.")
    parser.add_argument("--single", action="store_true", help="If set, process one random file per folder.")
    parser.add_argument("--case_id", type=str, default="", help="Specify a case ID (filename without extension) to process a single case.")
    parser.add_argument("--prompt_mode", type=str, default="", help="Optional suffix for your prompt JSON files.")
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
    os.setpgrp()  # Set process group for proper termination
    main()

# Usage example: single case with specific id

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt6" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"


# ==============================================================================================


# Usage example: single case with specific id

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt6" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"


# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt8" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1150440"


# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt9" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1178863"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt10" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"


# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt10" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"

# Usage example: single case with specific id

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt10" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt11" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"

# BEST ONE YET! mistake reverted! 
# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt12" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1130580"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt13" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1150440"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt19" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1178863"


# Still need to only save the summaries_text.txt time! 
# do ollama interactive and the one by one "what is???"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt15" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1150440"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt20" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1162274"

# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt21" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1181388"


# python hnc_reports_agent4.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt22" \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single \
#   --case_id "1211203"