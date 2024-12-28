#!/usr/bin/env python3
"""
Summarize Pathology & Consultation Notes
using line-by-line regex matching (re.MULTILINE) 
with anchored patterns (^FieldName: ...$).
Debug logs for pattern matching & DF shape,
but NOT printing the entire prompt or input text.
"""

import os
import json
import re
import argparse
import logging
import pickle
from typing import Dict, Any, Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# LangChain
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

# Model-specific
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture debug logs

##############################################################################
# 1. Fields Definitions
##############################################################################
PATHOLOGY_FIELDS = [
    "Age",
    "Sex",
    "Anatomic_Site_of_Lesion",
    "Cancer_Staging",
    "Pathological_TNM",
    "Clinical_TNM",
    "Primary_Tumor_Size",
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
    "Other_Tumor_Description"
]

CONSULTATION_FIELDS = [
    "Yes_No_Cannot_Infer",
    "Grade",
    "Tumor_Size",
    "SUV_from_PET_scans",
    "Pack_Years",
    "Patient_Concerns",
    "Recommendations",
    "Follow_up_Actions",
    "Alcohol_Consumption_Consult",
    "HPV_Status",
    "Charlson_Comorbidity_Score",
    "ECOG_Performance_Status",
    "Karnofsky_Performance_Status",
    "Patient_History_Status_Former_Status_of_Patient",
    "Patient_History_Status_Similar_Conditions",
    "Patient_History_Status_Previous_Treatments",
    "Clinical_Assessments_Radiological_Lesions",
    "Clinical_Assessments_SUV_from_PET_scans",
    "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score",
    "Performance_Comorbidity_Scores_ECOG_Performance_Status",
    "Performance_Comorbidity_Scores_Karnofsky_Performance_Status",
    "Cancer_Staging_Pathological_TNM",
    "Cancer_Staging_Clinical_TNM",
    "Cancer_Staging_Tumor_Size",
    "Others"
]

##############################################################################
# 2. Regex Patterns (Anchored at line start/end, re.MULTILINE)
##############################################################################
PATTERNS = {
    # -------------------------
    # Pathology (line-by-line)
    # -------------------------
    "Age": r"^Age:\s*(\d+|Not inferred)\s*$",
    "Sex": r"^Sex:\s*(Male|Female|Other|Not inferred)\s*$",
    "Anatomic_Site_of_Lesion": r"^Anatomic_Site_of_Lesion:\s*(.*)$",
    "Cancer_Staging": r"^Cancer_Staging:\s*(.*)$",
    "Pathological_TNM": r"^Pathological_TNM:\s*(.*)$",
    "Clinical_TNM": r"^Clinical_TNM:\s*(.*)$",
    "Primary_Tumor_Size": r"^Primary_Tumor_Size:\s*(.*)$",
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
    "Other_Tumor_Description": r"^Other_Tumor_Description:\s*(.*)$",

    # -------------------------
    # Consultation (line-by-line)
    # -------------------------
    "Yes_No_Cannot_Infer": r"^Yes_No_Cannot_Infer:\s*(Yes|No|Cannot Infer|Not inferred)\s*$",
    "Grade": r"^Grade:\s*(\d+|Not inferred)\s*$",
    "Tumor_Size": r"^Tumor_Size:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "SUV_from_PET_scans": r"^SUV_from_PET_scans:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "Pack_Years": r"^Pack_Years:\s*(\d+|Not inferred)\s*$",
    "Patient_Concerns": r"^Patient_Concerns:\s*(.*)$",
    "Recommendations": r"^Recommendations:\s*(.*)$",
    "Follow_up_Actions": r"^Follow_up_Actions:\s*(.*)$",
    "Alcohol_Consumption_Consult": r"^Alcohol_Consumption_Consult:\s*(Never drank|Ex-drinker|Drinker|Not inferred)\s*$",
    "HPV_Status": r"^HPV_Status:\s*(Positive|Negative|Not inferred)\s*$",
    "Charlson_Comorbidity_Score": r"^Charlson_Comorbidity_Score:\s*(\d+|Not inferred)\s*$",
    "ECOG_Performance_Status": r"^ECOG_Performance_Status:\s*(0|1|2|3|4|Not inferred)\s*$",
    "Karnofsky_Performance_Status": r"^Karnofsky_Performance_Status:\s*(100|90|80|70|60|50|40|30|20|10|0|Not inferred)\s*$",
    "Patient_History_Status_Former_Status_of_Patient": r"^Patient_History_Status_Former_Status_of_Patient:\s*(.*)$",
    "Patient_History_Status_Similar_Conditions": r"^Patient_History_Status_Similar_Conditions:\s*(.*)$",
    "Patient_History_Status_Previous_Treatments": r"^Patient_History_Status_Previous_Treatments:\s*(Radiation|Chemotherapy|Surgery|None|Not inferred)\s*$",
    "Clinical_Assessments_Radiological_Lesions": r"^Clinical_Assessments_Radiological_Lesions:\s*(.*)$",
    "Clinical_Assessments_SUV_from_PET_Scans": r"^Clinical_Assessments_SUV_from_PET_Scans:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score": r"^Performance_Comorbidity_Scores_Charlson_Comorbidity_Score:\s*(\d+|Not inferred)\s*$",
    "Performance_Comorbidity_Scores_ECOG_Performance_Status": r"^Performance_Comorbidity_Scores_ECOG_Performance_Status:\s*(0|1|2|3|4|Not inferred)\s*$",
    "Performance_Comorbidity_Scores_Karnofsky_Performance_Status": r"^Performance_Comorbidity_Scores_Karnofsky_Performance_Status:\s*(100|90|80|70|60|50|40|30|20|10|0|Not inferred)\s*$",
    "Cancer_Staging_Pathological_TNM": r"^Cancer_Staging_Pathological_TNM:\s*([T]\d+[N]\d+[M]\d+|Not inferred)\s*$",
    "Cancer_Staging_Clinical_TNM": r"^Cancer_Staging_Clinical_TNM:\s*([T]\d+[N]\d+[M]\d+|Not inferred)\s*$",
    "Cancer_Staging_Tumor_Size": r"^Cancer_Staging_Tumor_Size:\s*(\d+(\.\d+)?|Not inferred)\s*$",
    "Others": r"^Others:\s*(.*)$"
}

##############################################################################
# 3. Extraction & Validation
##############################################################################

def extract_tabular_data(summary: str, report_type: str) -> Dict[str, Any]:
    """Apply line-anchored regex for each field, using re.MULTILINE."""
    if report_type == "pathology_reports":
        fields = PATHOLOGY_FIELDS
    elif report_type == "consultation_notes":
        fields = CONSULTATION_FIELDS
    else:
        logger.warning(f"Unknown report type: {report_type}")
        return {}

    extracted = {}
    for field in fields:
        pattern = PATTERNS.get(field)
        if pattern:
            match = re.search(pattern, summary, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted_value = match.group(1).strip()
                logger.debug(f"[{report_type}] Matched '{field}': {extracted_value[:80]}...")  # Just a snippet
                extracted[field] = extracted_value
            else:
                logger.debug(f"[{report_type}] No match for '{field}'. Setting 'Not inferred'")
                extracted[field] = "Not inferred"
        else:
            logger.debug(f"[{report_type}] No pattern for '{field}'. Setting 'Not inferred'")
            extracted[field] = "Not inferred"
    return extracted

def validate_extracted_data(data: Dict[str, Any], report_type: str) -> bool:
    """Basic presence check."""
    if report_type == "pathology_reports":
        required_fields = PATHOLOGY_FIELDS
    elif report_type == "consultation_notes":
        required_fields = CONSULTATION_FIELDS
    else:
        return False

    for field in required_fields:
        if field not in data:
            logger.debug(f"[{report_type}] Missing field '{field}' in data.")
            return False
    return True

def normalize_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    """Strip whitespace, etc."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df

def encode_structured_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    """
    Example numeric/categorical encoding with logs to diagnose shape mismatches.
    """
    logger.debug(f"[{report_type}] Starting encode with shape: {df.shape}")
    df_encoded = df.copy()

    if report_type == "pathology_reports":
        cat_cols = [
            "Sex", "Anatomic_Site_of_Lesion", "Cancer_Staging", "Pathological_TNM",
            "Clinical_TNM", "Pathology_Details", "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Extranodal_Extension", "Resection_Margins", "p16_Status",
            "Immunohistochemical_profile", "EBER_Status", "Lymphovascular_Invasion_Status",
            "Perineural_Invasion_Status", "Other_Tumor_Description"
        ]
        num_cols = ["Age", "Lymph_Node_Status_Number_of_Positve_Lymph_Nodes"]

    elif report_type == "consultation_notes":
        cat_cols = [
            "Yes_No_Cannot_Infer", "Patient_Concerns", "Recommendations", "Follow_up_Actions",
            "Alcohol_Consumption_Consult", "HPV_Status", "Patient_History_Status_Former_Status_of_Patient",
            "Patient_History_Status_Similar_Conditions", "Patient_History_Status_Previous_Treatments",
            "Clinical_Assessments_Radiological_Lesions", "Others"
        ]
        num_cols = ["Grade", "Tumor_Size", "SUV_from_PET_scans", "Pack_Years"]
    else:
        logger.debug(f"[{report_type}] Unknown report type for encoding. Returning unmodified.")
        return df_encoded

    # Convert numeric
    for col in num_cols:
        if col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            logger.debug(f"[{report_type}] Numeric conversion '{col}' => {df_encoded[col].head(2).tolist()}")

    # Impute numeric
    if num_cols:
        imp = SimpleImputer(strategy='median')
        df_encoded[num_cols] = imp.fit_transform(df_encoded[num_cols])

    # Filter missing columns
    cat_cols_existing = [c for c in cat_cols if c in df_encoded.columns]
    if cat_cols_existing:
        enc = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        arr = enc.fit_transform(df_encoded[cat_cols_existing])
        encoded_df = pd.DataFrame(arr, columns=enc.get_feature_names_out(cat_cols_existing))
        df_encoded.drop(columns=cat_cols_existing, inplace=True)
        df_encoded.reset_index(drop=True, inplace=True)
        encoded_df.reset_index(drop=True, inplace=True)
        if len(df_encoded) != len(encoded_df):
            logger.debug(f"[{report_type}] Mismatch rows after encoding => df_encoded={len(df_encoded)}, encoded_df={len(encoded_df)}")
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    logger.debug(f"[{report_type}] Final shape after encoding: {df_encoded.shape}")
    return df_encoded

##############################################################################
# 4. Summarizer Class
##############################################################################

class ReportSummarizer:
    def __init__(
        self,
        prompts_dir: str,
        model_type: str = "local",
        temperature: float = 0.3,
        embedding_model: str = "ollama"
    ):
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.embedding_model = embedding_model.lower()

        # Load prompts
        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Invalid prompts_dir: {prompts_dir}")

        self.prompts = {}
        for rtype in ["pathology_reports", "consultation_notes"]:
            pfile = os.path.join(prompts_dir, f"prompt_{rtype}.json")
            if os.path.isfile(pfile):
                with open(pfile, 'r') as f:
                    data = json.load(f)
                    self.prompts[rtype] = data.get("prompts", [])
                logger.info(f"Loaded prompts for {rtype}")
            else:
                logger.warning(f"No prompt file found for {rtype}")
                self.prompts[rtype] = []

        # Initialize LLM
        if self.model_type == "local":
            self.model = ChatOllama(model="llama3.3:latest", temperature=self.temperature)
        elif self.model_type == "gpt":
            self.model = ChatOpenAI(model="gpt-4", temperature=self.temperature)
        elif self.model_type == "gemini":
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize Embeddings
        if self.embedding_model == "ollama":
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        elif self.embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif self.embedding_model == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        else:
            logger.warning(f"Unknown embedding_model: {embedding_model}. Defaulting to Ollama.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.chain_map = {}
        for rtype, prompt_list in self.prompts.items():
            if not prompt_list:
                continue
            prompt_text = "\n".join(prompt_list)
            self.chain_map[rtype] = self.make_llm_runnable(prompt_text)

    def make_llm_runnable(self, prompt: str) -> RunnableLambda:
        def llm_runnable(inputs: Dict[str, str]) -> Dict[str, str]:
            # We'll avoid printing the entire final_prompt or input text
            # Just log a short message
            context_len = len(inputs["context"])
            logger.debug(f"Summarizing a {context_len}-char input for a specific report_type (prompt not fully shown).")
            try:
                result = self.model.invoke([HumanMessage(content=prompt.replace("{context}", inputs["context"]))]).content
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
        resp = runnable.invoke({"context": report_text})
        summary = resp.get("summary", "").strip()
        logger.debug(f"[{report_type}] Summarize output (first 80 chars): {summary[:80]}...")
        return summary if summary else None

    def process_reports(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        summaries = []
        tabular_data = []
        invalid_entries = []

        # Typically subfolders named "pathology_reports" or "consultation_notes"
        report_type_map = {
            "pathology_reports": "pathology_reports",
            "consultation_notes": "consultation_notes"
        }

        for root, dirs, files in os.walk(input_dir):
            parent = os.path.basename(root).lower()
            rtype = report_type_map.get(parent, None)

            for fname in files:
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(root, fname)
                logger.info(f"Processing file: {fname} in folder: {parent}")

                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    if not rtype:
                        invalid_entries.append({"file": fname, "reason": "Unknown folder for report type"})
                        logger.warning(f"Skipping {fname}, unknown folder {parent}")
                        continue

                    # Summarize
                    summary = self.summarize_report(text, rtype)
                    if not summary:
                        invalid_entries.append({"file": fname, "reason": "No summary produced"})
                        logger.warning(f"Failed to generate summary for {fname}")
                        continue

                    # Extract line-by-line
                    extracted = extract_tabular_data(summary, rtype)
                    if not validate_extracted_data(extracted, rtype):
                        invalid_entries.append({"file": fname, "reason": "Validation failed"})
                        logger.debug(f"[{rtype}] Validation failed for {fname}")

                    # Keep track
                    summaries.append({"file": fname, "report_type": rtype, "summary": summary})
                    tabular_data.append({"file": fname, "report_type": rtype, "data": extracted})

                    # Prepare output subfolders
                    patient_id = os.path.splitext(fname)[0]
                    subdirs = {
                        "text_summaries": os.path.join(output_dir, "text_summaries", rtype, patient_id),
                        "embeddings": os.path.join(output_dir, "embeddings", rtype, patient_id),
                        "structured_data": os.path.join(output_dir, "structured_data", rtype, patient_id),
                        "structured_data_encoded": os.path.join(output_dir, "structured_data_encoded", rtype, patient_id)
                    }
                    for sd in subdirs.values():
                        os.makedirs(sd, exist_ok=True)

                    # Save summary
                    summary_path = os.path.join(subdirs["text_summaries"], f"{rtype}_summary.txt")
                    with open(summary_path, 'w', encoding='utf-8') as sf:
                        sf.write(summary)
                    logger.info(f"Summary saved to {summary_path}")

                    # Embeddings
                    emb_path = os.path.join(subdirs["embeddings"], f"{rtype}_embedding.pkl")
                    try:
                        emb = self.embeddings.embed_documents([summary])[0]
                        with open(emb_path, 'wb') as ef:
                            pickle.dump(emb, ef)
                        logger.info(f"Embedding saved to {emb_path}")
                    except Exception as e:
                        invalid_entries.append({"file": fname, "reason": f"Embedding error: {e}"})
                        logger.error(f"Embedding error for {fname}: {e}")

                    # Structured CSV
                    df_struct = pd.DataFrame([extracted])
                    df_struct = normalize_data(df_struct, rtype)
                    struct_path = os.path.join(subdirs["structured_data"], f"{rtype}_structured.csv")
                    logger.debug(f"Saving structured data to {struct_path}, shape: {df_struct.shape}")
                    df_struct.to_csv(struct_path, index=False)
                    logger.info(f"Structured data saved to {struct_path}")

                    # Encoded CSV
                    df_encoded = encode_structured_data(df_struct, rtype)
                    encoded_path = os.path.join(subdirs["structured_data_encoded"], f"{rtype}_structured_encoded.csv")
                    logger.debug(f"Saving encoded data to {encoded_path}, shape: {df_encoded.shape}")
                    df_encoded.to_csv(encoded_path, index=False)
                    logger.info(f"Encoded structured data saved to {encoded_path}")

                except Exception as e:
                    invalid_entries.append({"file": fname, "reason": str(e)})
                    logger.error(f"Error processing {fname}: {e}")

        # Summaries metadata
        if summaries:
            summaries_csv = os.path.join(output_dir, "summaries_metadata.csv")
            pd.DataFrame(summaries).to_csv(summaries_csv, index=False)
            logger.info(f"Summaries metadata saved to {summaries_csv}")

        # Tabular data metadata
        if tabular_data:
            tabular_csv = os.path.join(output_dir, "tabular_data_metadata.csv")
            pd.DataFrame(tabular_data).to_csv(tabular_csv, index=False)
            logger.info(f"Tabular data metadata saved to {tabular_csv}")

        # Invalid
        if invalid_entries:
            invalid_log = os.path.join(output_dir, "invalid_entries.log")
            pd.DataFrame(invalid_entries).to_csv(invalid_log, index=False)
            logger.info(f"Invalid entries logged to {invalid_log}")

def main():
    parser = argparse.ArgumentParser(description="Summarize HNC Pathology & Consultation Notes, with debug logs & minimal prompt printing.")
    parser.add_argument("--prompts_dir", required=True, help="Directory with JSON prompt files (prompt_pathology_reports.json, prompt_consultation_notes.json)")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend to use")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for LLM responses")
    parser.add_argument("--input_dir", required=True, help="Directory with subfolders: pathology_reports/ and consultation_notes/")
    parser.add_argument("--output_dir", required=True, help="Output directory for summaries, embeddings, CSVs")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="ollama",
        choices=["ollama","openai","google"],
        help="Embedding model to use (ollama, openai, google)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summarizer = ReportSummarizer(
        prompts_dir=args.prompts_dir,
        model_type=args.model_type,
        temperature=args.temperature,
        embedding_model=args.embedding_model
    )
    summarizer.process_reports(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
