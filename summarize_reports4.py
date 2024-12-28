#!/usr/bin/env python3
"""
Summarize Pathology & Consultation Notes
1) Structured Summary (line-by-line)
2) Unstructured (paragraph) Summary
Stores both text outputs, both embeddings, and logs processing times.
Replaces "Not inferred" with NaN for structured data, one-hot encodes, etc.
"""

import os
import json
import re
import argparse
import logging
import pickle
from typing import Dict, Any, Optional
import time
import numpy as np
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
logger.setLevel(logging.DEBUG)

##############################################################################
# 1. Field Definitions
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
    "Smoking_History",
    "Alcohol_Consumption",
    "Pack_Years",
    "Patient_Concerns",
    "Recommendations",
    "Follow_up_Actions",
    "HPV_Status",
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
# 2. Regex Patterns for Structured Summaries
##############################################################################
PATTERNS = {
    # (Same as your existing line-by-line anchored patterns)
    "Age": r"^Age:\s*(\d+|Not inferred)\s*$",
    # ... etc ...
    # (Truncated for brevity—keep the same patterns you already have)
}

##############################################################################
# 3. Extraction, Validation, Normalization, Encoding (Same as current)
##############################################################################
def extract_tabular_data(summary: str, report_type: str) -> Dict[str, Any]:
    # same as your code
    ...

def validate_extracted_data(data: Dict[str, Any], report_type: str) -> bool:
    # same as your code
    ...

def normalize_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    # same as your code (replace "Not inferred" with np.nan, etc.)
    ...

def encode_structured_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    # same as your code
    ...

##############################################################################
# 4. Summarizer Class (Modified to handle two sets of prompts)
##############################################################################
class ReportSummarizer:
    """
    Now we load TWO sets of prompts for each type:
    1) The "structured" prompts (original).
    2) The "unstructured" prompts (new).
    We store them in self.chain_map (structured) and self.chain_map_unstructured (unstructured).
    """

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

        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Invalid prompts_dir: {prompts_dir}")

        # We'll load two sets of prompts:
        #   prompt_<rtype>.json => structured
        #   prompt_<rtype>_unstructured.json => unstructured
        # for rtype in ["pathology_reports", "consultation_notes"]
        self.prompts_structured = {}
        self.prompts_unstructured = {}

        for rtype in ["pathology_reports", "consultation_notes"]:
            # structured
            file_struct = os.path.join(prompts_dir, f"prompt_{rtype}.json")
            if os.path.isfile(file_struct):
                with open(file_struct, 'r') as f:
                    data = json.load(f)
                    self.prompts_structured[rtype] = data.get("prompts", [])
                logger.info(f"Loaded structured prompts for {rtype}")
            else:
                logger.warning(f"No structured prompt file found for {rtype}")
                self.prompts_structured[rtype] = []

            # unstructured
            file_unstruct = os.path.join(prompts_dir, f"prompt_{rtype}_unstructured.json")
            if os.path.isfile(file_unstruct):
                with open(file_unstruct, 'r') as f:
                    data = json.load(f)
                    self.prompts_unstructured[rtype] = data.get("prompts", [])
                logger.info(f"Loaded unstructured prompts for {rtype}")
            else:
                logger.warning(f"No unstructured prompt file found for {rtype}")
                self.prompts_unstructured[rtype] = []

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

        # Build chain maps: structured + unstructured
        self.chain_map_structured = {}
        self.chain_map_unstructured = {}

        for rtype in ["pathology_reports", "consultation_notes"]:
            # structured
            p_struct = self.prompts_structured[rtype]
            if p_struct:
                prompt_text_struct = "\n".join(p_struct)
                self.chain_map_structured[rtype] = self.make_llm_runnable(prompt_text_struct)
            # unstructured
            p_unstruct = self.prompts_unstructured[rtype]
            if p_unstruct:
                prompt_text_unstruct = "\n".join(p_unstruct)
                self.chain_map_unstructured[rtype] = self.make_llm_runnable(prompt_text_unstruct)

    def make_llm_runnable(self, prompt: str) -> RunnableLambda:
        def llm_runnable(inputs: Dict[str, str]) -> Dict[str, str]:
            context = inputs["context"]
            logger.debug(f"Summarizing text len={len(context)} for {inputs.get('report_type', 'unknown')}")
            try:
                final_prompt = prompt.replace("{context}", context)
                response = self.model.invoke([HumanMessage(content=final_prompt)]).content
                return {"summary": response}
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return {"summary": ""}
        return RunnableLambda(llm_runnable)

    def summarize_report_structured(self, report_text: str, report_type: str) -> Optional[str]:
        """Generate a structured summary (line-by-line) if we have a structured chain."""
        if report_type not in self.chain_map_structured:
            logger.warning(f"No structured prompt found for {report_type}")
            return None
        runnable = self.chain_map_structured[report_type]
        resp = runnable.invoke({"context": report_text, "report_type": report_type})
        summary = resp.get("summary", "").strip()
        logger.debug(f"[{report_type}] Structured summary (80 chars): {summary[:80]}")
        return summary if summary else None

    def summarize_report_unstructured(self, report_text: str, report_type: str) -> Optional[str]:
        """Generate an unstructured (paragraph) summary if we have an unstructured chain."""
        if report_type not in self.chain_map_unstructured:
            logger.warning(f"No unstructured prompt found for {report_type}")
            return None
        runnable = self.chain_map_unstructured[report_type]
        resp = runnable.invoke({"context": report_text, "report_type": report_type})
        summary = resp.get("summary", "").strip()
        logger.debug(f"[{report_type}] Unstructured summary (80 chars): {summary[:80]}")
        return summary if summary else None

    def process_reports(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        summaries = []      # store structured summary metadata
        tabular_data = []
        invalid_entries = []
        time_data = []      # store times

        # We'll also keep separate "unstructured_summaries" if you want
        unstructured_summaries_list = []

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
                start_time = time.time()

                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    if not rtype:
                        invalid_entries.append({"file": fname, "reason": "Unknown folder for report type"})
                        logger.warning(f"Skipping {fname}, unknown folder {parent}")
                        continue

                    # 1) Structured summary
                    structured_summary = self.summarize_report_structured(text, rtype)
                    # 2) Unstructured summary
                    unstructured_summary = self.summarize_report_unstructured(text, rtype)

                    if not structured_summary and not unstructured_summary:
                        invalid_entries.append({"file": fname, "reason": "No summary produced"})
                        logger.warning(f"No structured or unstructured summary for {fname}")
                        continue

                    # If we got a structured summary => extract tabular data
                    extracted = {}
                    if structured_summary:
                        extracted = extract_tabular_data(structured_summary, rtype)
                        if not validate_extracted_data(extracted, rtype):
                            invalid_entries.append({"file": fname, "reason": "Validation failed"})
                            logger.debug(f"[{rtype}] Validation failed for {fname}")

                    # Record structured summary in 'summaries' list
                    if structured_summary:
                        summaries.append({
                            "file": fname,
                            "report_type": rtype,
                            "summary": structured_summary
                        })
                    # Also record unstructured summary in 'unstructured_summaries_list'
                    if unstructured_summary:
                        unstructured_summaries_list.append({
                            "file": fname,
                            "report_type": rtype,
                            "summary_unstructured": unstructured_summary
                        })

                    if extracted:
                        tabular_data.append({"file": fname, "report_type": rtype, "data": extracted})

                    # Prepare output subfolders
                    patient_id = os.path.splitext(fname)[0]
                    subdirs = {
                        "text_summaries": os.path.join(output_dir, "text_summaries", rtype, patient_id),
                        "embeddings": os.path.join(output_dir, "embeddings", rtype, patient_id),
                        "structured_data": os.path.join(output_dir, "structured_data", rtype, patient_id),
                        "structured_data_encoded": os.path.join(output_dir, "structured_data_encoded", rtype, patient_id),
                        # For unstructured:
                        "tex_summ_unstructured": os.path.join(output_dir, "tex_summ_unstructured", rtype, patient_id),
                        "embeddings_tex_summ_unstructured": os.path.join(output_dir, "embeddings_tex_summ_unstructured", rtype, patient_id)
                    }
                    for sd in subdirs.values():
                        os.makedirs(sd, exist_ok=True)

                    # ----- Save structured summary if present -----
                    if structured_summary:
                        summary_path = os.path.join(subdirs["text_summaries"], f"{rtype}_summary.txt")
                        try:
                            with open(summary_path, 'w', encoding='utf-8') as sf:
                                sf.write(structured_summary)
                            logger.info(f"Structured summary saved to {summary_path}")
                        except Exception as e:
                            invalid_entries.append({"file": fname, "reason": f"Failed to save structured summary: {e}"})
                            logger.error(f"Could not save structured summary for {fname}: {e}")

                        # Embeddings for structured summary
                        emb_path = os.path.join(subdirs["embeddings"], f"{rtype}_embedding.pkl")
                        try:
                            emb = self.embeddings.embed_documents([structured_summary])[0]
                            with open(emb_path, 'wb') as ef:
                                pickle.dump(emb, ef)
                            logger.info(f"Structured embedding saved to {emb_path}")
                        except Exception as e:
                            invalid_entries.append({"file": fname, "reason": f"Embedding error (structured): {e}"})
                            logger.error(f"Embedding error for {fname} (structured): {e}")

                        # Save structured CSV
                        if extracted:
                            df_struct = pd.DataFrame([extracted])
                            df_struct = normalize_data(df_struct, rtype)
                            struct_path = os.path.join(subdirs["structured_data"], f"{rtype}_structured.csv")
                            logger.debug(f"Saving structured data to {struct_path}, shape: {df_struct.shape}")
                            try:
                                df_struct.to_csv(struct_path, index=False)
                                logger.info(f"Structured data saved to {struct_path}")
                            except Exception as e:
                                invalid_entries.append({"file": fname, "reason": f"Failed to save structured CSV: {e}"})
                                logger.error(f"Could not save structured CSV for {fname}: {e}")
                                continue

                            # Encoded CSV
                            df_encoded = encode_structured_data(df_struct, rtype)
                            encoded_path = os.path.join(subdirs["structured_data_encoded"], f"{rtype}_structured_encoded.csv")
                            logger.debug(f"Saving encoded data to {encoded_path}, shape: {df_encoded.shape}")
                            try:
                                df_encoded.to_csv(encoded_path, index=False)
                                logger.info(f"Encoded structured data saved to {encoded_path}")
                            except Exception as e:
                                invalid_entries.append({"file": fname, "reason": f"Failed to save encoded CSV: {e}"})
                                logger.error(f"Could not save encoded CSV for {fname}: {e}")

                    # ----- Save unstructured summary if present -----
                    if unstructured_summary:
                        unstruct_path = os.path.join(subdirs["tex_summ_unstructured"], f"{rtype}_summary_unstructured.txt")
                        try:
                            with open(unstruct_path, 'w', encoding='utf-8') as uf:
                                uf.write(unstructured_summary)
                            logger.info(f"Unstructured summary saved to {unstruct_path}")
                        except Exception as e:
                            invalid_entries.append({"file": fname, "reason": f"Failed to save unstructured summary: {e}"})
                            logger.error(f"Could not save unstructured summary for {fname}: {e}")

                        # Embeddings for unstructured summary
                        emb_unstruct_path = os.path.join(subdirs["embeddings_tex_summ_unstructured"], f"{rtype}_embedding_unstructured.pkl")
                        try:
                            emb_unstruct = self.embeddings.embed_documents([unstructured_summary])[0]
                            with open(emb_unstruct_path, 'wb') as ef:
                                pickle.dump(emb_unstruct, ef)
                            logger.info(f"Unstructured embedding saved to {emb_unstruct_path}")
                        except Exception as e:
                            invalid_entries.append({"file": fname, "reason": f"Embedding error (unstructured): {e}"})
                            logger.error(f"Embedding error for {fname} (unstructured): {e}")

                except Exception as e:
                    invalid_entries.append({"file": fname, "reason": str(e)})
                    logger.error(f"Error processing {fname}: {e}")

                end_time = time.time()
                process_time = end_time - start_time
                time_data.append({
                    "file": fname,
                    "report_type": rtype if rtype else "unknown",
                    "process_time_seconds": round(process_time, 3)
                })
                logger.info(f"Processed {fname} in {round(process_time, 3)} seconds.")

        # Save Summaries Metadata
        if summaries:
            summaries_csv = os.path.join(output_dir, "summaries_metadata.csv")
            try:
                pd.DataFrame(summaries).to_csv(summaries_csv, index=False)
                logger.info(f"Summaries metadata saved to {summaries_csv}")
            except Exception as e:
                logger.error(f"Failed to save summaries metadata: {e}")

        # If you also want to store unstructured_summaries metadata:
        if unstructured_summaries_list:
            unstruct_csv = os.path.join(output_dir, "unstructured_summaries_metadata.csv")
            try:
                pd.DataFrame(unstructured_summaries_list).to_csv(unstruct_csv, index=False)
                logger.info(f"Unstructured summaries metadata saved to {unstruct_csv}")
            except Exception as e:
                logger.error(f"Failed to save unstructured summaries metadata: {e}")

        # Tabular data metadata
        if tabular_data:
            tabular_csv = os.path.join(output_dir, "tabular_data_metadata.csv")
            try:
                pd.DataFrame(tabular_data).to_csv(tabular_csv, index=False)
                logger.info(f"Tabular data metadata saved to {tabular_csv}")
            except Exception as e:
                logger.error(f"Failed to save tabular data metadata: {e}")

        # Invalid
        if invalid_entries:
            invalid_log = os.path.join(output_dir, "invalid_entries.log")
            try:
                pd.DataFrame(invalid_entries).to_csv(invalid_log, index=False)
                logger.info(f"Invalid entries logged to {invalid_log}")
            except Exception as e:
                logger.error(f"Failed to save invalid entries log: {e}")

        # Save timing data
        times_csv = os.path.join(output_dir, "processing_times.csv")
        try:
            df_times = pd.DataFrame(time_data)
            if os.path.isfile(times_csv):
                existing_times = pd.read_csv(times_csv)
                combined = pd.concat([existing_times, df_times], ignore_index=True)
                combined.to_csv(times_csv, index=False)
            else:
                df_times.to_csv(times_csv, index=False)
            logger.info(f"Processing times saved/updated in {times_csv}")
        except Exception as e:
            logger.error(f"Failed to save processing_times CSV: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Pathology & Consultation Notes, generating both structured & unstructured outputs."
    )
    parser.add_argument("--prompts_dir", required=True, help="Directory containing all prompt JSON files.")
    parser.add_argument("--model_type", default="local", choices=["local","gpt","gemini"], help="LLM backend to use.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for LLM responses.")
    parser.add_argument("--input_dir", required=True, help="Directory with subfolders: pathology_reports/ and consultation_notes/.")
    parser.add_argument("--output_dir", required=True, help="Output directory for summaries, embeddings, CSVs.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="ollama",
        choices=["ollama","openai","google"],
        help="Embedding model to use (ollama, openai, google)."
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
