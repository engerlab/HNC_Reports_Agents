#!/usr/bin/env python3
"""
Summarize HNC Reports:
  - Processes PathologyReports/ and ConsultRedacted/ from their respective subfolders.
  - Also processes combined reports from PathConsCombined/ in three different modes:
      • treatment_plan_outcomepred (original prompt)
      • path_consult_reports (combined pathology+consultation extraction)
      • cot_treatment_plan_outcomepred (chain-of-thought treatment plan prompt)

Usage:
  --report_type: Comma-separated list of report types to process. Options:
      pathology_reports,
      consultation_notes,
      treatment_plan_outcomepred,
      path_consult_reports,
      cot_treatment_plan_outcomepred.
      Default is "all".
  --local_model: If using local model, specify the model name (default: "llama3.3:latest").
  --prompt_mode: (Optional) Specify the prompt version suffix (e.g., "combined" or leave empty for default).
  --single: (Optional flag) If set, process one random file per folder.
  --case_id: (Optional) Specify a case ID (filename without extension) to process a specific case.
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

# Model-specific
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
    "Other_diagnostic_finding"
]

CONSULTATION_FIELDS = [
    "Smoking_History",
    "Alcohol_Consumption",
    "Pack_Years",
    "Patient_Symptoms_at_Presentation",
    "Recommendations",
    "Plans",
    "HPV_Status",
    "Patient_History_Status_Prior_Conditions",
    "Patient_History_Status_Previous_Treatments",
    "Clinical_Assessments_Radiological_Lesions",
    "Clinical_Assessments_SUV_from_PET_scans",
    "Charlson_Comorbidity_Score",
    "Karnofsky_Performance_Status",
    "ECOG_Performance_Status",


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
    "Other_diagnostic_finding": r"^Other_diagnostic_finding:\s*(.*)$",
    "Smoking_History": r"^Smoking_History:\s*(.*)$",
    "Alcohol_Consumption": r"^Alcohol_Consumption:\s*(.*)$",
    "Pack_Years": r"^Pack_Years:\s*(\d+|Not inferred)\s*$",
    "Patient_Symptoms_at_Presentation": r"^Patient_Symptoms_at_Presentation:\s*(.*)$",
    "Recommendations": r"^Recommendations:\s*(.*)$",
    "Plans": r"^Plans:\s*(.*)$",
    "HPV_Status": r"^HPV_Status:\s*(Positive|Negative|Not inferred)\s*$",
    "Patient_History_Status_Prior_Conditions": r"^Patient_History_Status_Prior_Conditions:\s*(.*)$",
    "Patient_History_Status_Previous_Treatments": r"^Patient_History_Status_Previous_Treatments:\s*(.*)$",
    "Clinical_Assessments_Radiological_Lesions": r"^Clinical_Assessments_Radiological_Lesions:\s*(.*)$",
    "Clinical_Assessments_SUV_from_PET_scans": r"^Clinical_Assessments_SUV_from_PET_scans:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "Charlson_Comorbidity_Score": r"^Charlson_Comorbidity_Score:\s*(\d+|Not inferred)\s*$",
    "Karnofsky_Performance_Status": r"^Karnofsky_Performance_Status:\s*(100|90|80|70|60|50|40|30|20|10|0|Not inferred)\s*$",
    "ECOG_Performance_Status": r"^ECOG_Performance_Status:\s*(0|1|2|3|4|)\s*$"
}

##############################################################################
# 3. Extraction & Validation Functions
##############################################################################
def extract_tabular_data(summary: str, report_type: str) -> Dict[str, Any]:
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
            df[col] = df[col].str.strip()
    return df

def encode_structured_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    logger.debug(f"[{report_type}] Starting encode with shape: {df.shape}")
    df_encoded = df.copy()
    if report_type == "pathology_reports":
        cat_cols = [
            "Sex", "Anatomic_Site_of_Lesion", "Pathological_TNM",
            "Clinical_TNM", "Pathology_Details", "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Extranodal_Extension", "Resection_Margins", "p16_Status",
            "Immunohistochemical_profile", "EBER_Status", "Lymphovascular_Invasion_Status",
            "Perineural_Invasion_Status", "Other_diagnostic_finding"
        ]
        num_cols = ["Age", "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes"]
    elif report_type == "consultation_notes":
        cat_cols = [
            "Smoking_History", "Alcohol_Consumption", "Patient_Symptoms_at_Presentation",
            "Recommendations", "Plans", "HPV_Status", "Patient_History_Status_Prior_Conditions",
            "Patient_History_Status_Previous_Treatments", "Clinical_Assessments_Radiological_Lesions"
        ]
        num_cols = ["Pack_Years", "Clinical_Assessments_SUV_from_PET_scans", "Charlson_Comorbidity_Score"]
        num_cols += ["Karnofsky_Performance_Status", "ECOG_Performance_Status"]
    elif report_type == "path_consult_reports":
        cat_cols = [
            "Sex", "Anatomic_Site_of_Lesion", "Pathological_TNM",
            "Clinical_TNM", "Pathology_Details", "Tumor_Type_Differentiation", "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes", "Lymph_Node_Status_Extranodal_Extension",
            "Resection_Margins", "p16_Status", "Immunohistochemical_profile", "EBER_Status",
            "Lymphovascular_Invasion_Status", "Perineural_Invasion_Status", "Other_diagnostic_finding",
            "Smoking_History", "Alcohol_Consumption", "Patient_Symptoms_at_Presentation",
            "Recommendations", "Plans", "HPV_Status", "Patient_History_Status_Prior_Conditions",
            "Patient_History_Status_Previous_Treatments", "Clinical_Assessments_Radiological_Lesions",
            "Clinical_Assessments_SUV_from_PET_scans", "Charlson_Comorbidity_Score",
            "Karnofsky_Performance_Status", "ECOG_Performance_Status"
        ]
        num_cols = []
    else:
        logger.debug(f"[{report_type}] Unknown type for encoding. Returning unmodified.")
        return df_encoded

    for col in num_cols:
        if col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            logger.debug(f"[{report_type}] Numeric conversion '{col}': {df_encoded[col].head(2).tolist()}")
    if num_cols:
        imp = SimpleImputer(strategy='median')
        df_encoded[num_cols] = imp.fit_transform(df_encoded[num_cols])
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
    Given a summary string (generated by the LLM) and a list of expected field names,
    enforce that the output contains exactly these fields in order.
    For each expected field, if a line starting with "FieldName:" is present in the summary,
    use that line; otherwise, output "FieldName: Not inferred".
    """
    lines = summary.splitlines()
    field_lines = {}
    for field in expected_fields:
        found_line = None
        pattern = f"^{field}:"  # match field name exactly
        for line in lines:
            if re.match(pattern, line):
                found_line = line.strip()
                break
        if found_line:
            field_lines[field] = found_line
        else:
            field_lines[field] = f"{field}: Not inferred"
    # Return the lines in the order of expected_fields
    return "\n".join([field_lines[field] for field in expected_fields])

##############################################################################
# 4. Summarizer Class
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

        # Load prompts for each known report type.
        # If prompt_mode is specified, attempt to load prompt files with that suffix.
        self.prompts = {}
        for rtype in [
            "pathology_reports",
            "consultation_notes",
            "treatment_plan_outcomepred",
            "path_consult_reports",
            "cot_treatment_plan_outcomepred"
        ]:
            filename = f"prompt_{rtype}"
            if self.prompt_mode:
                candidate = os.path.join(prompts_dir, f"{filename}_{self.prompt_mode}.json")
                if os.path.isfile(candidate):
                    pfile = candidate
                else:
                    pfile = os.path.join(prompts_dir, f"{filename}.json")
            else:
                pfile = os.path.join(prompts_dir, f"{filename}.json")
            if os.path.isfile(pfile):
                with open(pfile, 'r') as f:
                    data = json.load(f)
                    self.prompts[rtype] = data.get("prompts", [])
                logger.info(f"Loaded prompts for {rtype} from {pfile}")
            else:
                logger.warning(f"No prompt file found for {rtype} (expected {pfile})")
                self.prompts[rtype] = []

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
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        else:
            logger.warning(f"Unknown embedding_model: {self.embedding_model}. Defaulting to Ollama.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build a map of rtype -> prompt-based function
        self.chain_map = {}
        for rtype, prompt_list in self.prompts.items():
            if not prompt_list:
                continue
            prompt_text = "\n".join(prompt_list)
            self.chain_map[rtype] = self.make_llm_runnable(prompt_text)

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

    def summarize_report(self, report_text: str, report_type: str) -> Optional[str]:
        if report_type not in self.chain_map:
            logger.warning(f"No prompt found for {report_type}")
            return None
        runnable = self.chain_map[report_type]
        resp = runnable.invoke({"context": report_text, "report_type": report_type})
        summary = resp.get("summary", "").strip()
        logger.debug(f"[{report_type}] Summarize output (first 80 chars): {summary[:80]}...")
        # For structured fields, enforce the format
        if report_type in ["pathology_reports", "consultation_notes", "path_consult_reports"]:
            if report_type == "pathology_reports":
                expected = PATHOLOGY_FIELDS
            elif report_type == "consultation_notes":
                expected = CONSULTATION_FIELDS
            else:
                expected = PATH_CONS_FIELDS
            summary = enforce_format(summary, expected)
        return summary if summary else None

    def process_reports(self, input_dir: str, output_dir: str, report_types: List[str], single: bool = False, case_id: Optional[str] = None):
        """
        - 'pathology_reports' => read from PathologyReports/ in input_dir
        - 'consultation_notes' => read from ConsultRedacted/ in input_dir
        - 'treatment_plan_outcomepred', 'path_consult_reports', 'cot_treatment_plan_outcomepred'
          => read from PathConsCombined/ in input_dir.
        If single is True, process only one file per folder.
        If case_id is provided, process only the file whose basename (without extension) matches that ID.
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
                    # Compute additional metrics
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
                    with open(os.path.join(subdirs["text_summaries"], f"{rtype}_summary.txt"), 'w', encoding='utf-8') as sf:
                        sf.write(summary)
                    emb = self.embeddings.embed_documents([summary])[0]
                    with open(os.path.join(subdirs["embeddings"], f"{rtype}_embedding.pkl"), 'wb') as ef:
                        pickle.dump(emb, ef)
                    # For structured outputs, only save processing_times.csv (omit other metadata CSVs)
                    df_struct = pd.DataFrame()  # We are not saving structured CSVs now.
                    df_encoded = pd.DataFrame()
                except Exception as e:
                    logger.error(f"Error processing {fname}: {e}")
        try:
            pd.DataFrame(time_data).to_csv(os.path.join(output_dir, "processing_times.csv"), index=False)
            logger.info(f"Processing times saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save processing_times.csv: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize HNC Reports. Input directory must contain:\n"
                    "  PathologyReports/  => for 'pathology_reports'\n"
                    "  ConsultRedacted/   => for 'consultation_notes'\n"
                    "  PathConsCombined/  => for 'treatment_plan_outcomepred', 'path_consult_reports', 'cot_treatment_plan_outcomepred'."
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
    parser.add_argument("--case_id", type=str, default="", help="Optional: specify a case ID (filename without extension) to process a specific case.")
    parser.add_argument("--prompt_mode", type=str, default="", help="Optional prompt mode (e.g., 'combined' or 'separated').")
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


# Usage example: (Full Processing)
# python hnc_reports_agent3.py --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts --model_type local --temperature 0.8 --input_dir "/media/yujing/One Touch3/HNC_Reports" --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpFull1" --embedding_model ollama --report_type "path_consult_reports" --local_model "llama3.3:latest" --prompt_mode "combined"

# To Process One Random Case (per subfolder):
# python hnc_reports_agent3.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPrompt \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --single # do one single random case 

# To Process a Specific Case by its Case ID:

# python hnc_reports_agent3.py \
#   --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
#   --model_type local \
#   --temperature 0.8 \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/ExpPromptsEng/ExpPrompt5 \
#   --embedding_model ollama \
#   --report_type "path_consult_reports" \
#   --local_model "llama3.3:latest" \
#   --prompt_mode "combined" \
#   --case_id "1130580"
