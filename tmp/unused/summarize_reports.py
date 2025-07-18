#!/usr/bin/env python3
"""
Pathology Report and Consultation Notes Summarizer using Llama 3.3 and LangChain

This script reads pathology report and consultation note text files organized in separate
subdirectories, generates concise summaries with key information using the specified model,
performs quality control, generates embeddings for the summaries, converts these summaries
into structured tabular data, and encodes the data for survival analysis models.

It outputs:
1. Text summaries and their embeddings for each report type.
2. Structured data tables for each report type.
3. Encoded structured data tables suitable for survival analysis.
"""

import os
import json
import re
import argparse
import logging
import pickle
from typing import List, Dict, Any, Optional
import random

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# LangChain and Model-specific Imports
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import RegexParser
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage

# Model-specific imports (ensure these are installed and correctly referenced)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Regular Expressions for Post-Processing
PATTERNS = {
    # Consultation Notes
    "Yes/No/Cannot Infer": r"Categorical Data:\s*(Yes|No|Cannot Infer)",
    "Grade": r"Grades?:\s*(Grade\s*\d+)",
    "Tumor Size": r"Tumor Size\s*\(mm\):\s*(\d+\.?\d*)",
    "SUV from PET scans": r"SUV from PET scans\s*:\s*(\d+\.?\d*)",
    "Pack Years": r"Pack Years\s*:\s*(\d+)",
    "Alcohol Consumption": r"Alcohol Consumption\s*:\s*(Never drank|Ex-drinker|Drinker)",
    "HPV Status": r"HPV Status\s*:\s*(Positive|Negative)",
    "Charlson Comorbidity Score": r"Charlson Comorbidity Score\s*:\s*(\d+)",
    "ECOG Performance Status": r"ECOG Performance Status\s*:\s*(\d+)",
    "Karnofsky Performance Status": r"Karnofsky Performance Status\s*:\s*(\d+)",

    # Pathology Reports
    "Clinical TNM Staging": r"Clinical TNM Staging\s*:\s*([T]\d+[N]\d+[M]\d+)",
    "Pathological TNM Staging": r"Pathological TNM Staging\s*:\s*([T]\d+[N]\d+[M]\d+)",
    "Lymph Node Status": r"Lymph Node Status\s*:\s*(Presence|Absence),?\s*(?:Number of Lymph Nodes\s*:\s*(\d+))?,?\s*(?:Extranodal Extension\s*:\s*(Yes|No))?",
    "Resection Margins": r"Resection Margins\s*:\s*(Positive|Negative)",
    "Biomarkers": r"Biomarkers?:\s*(.*)",

    # General Patterns
    "Diagnosis": r"Diagnosis:\s*(.*)",
    "Patient Concerns": r"Patient Concerns:\s*(.*)",
    "Recommendations": r"Recommendations:\s*(.*)",
    "Follow-up Actions": r"Follow-up Actions:\s*(.*)",

    # Others
    "Others": r"Others\s*:\s*(.*)"
}

def extract_tabular_data(summary: str, report_type: str) -> Dict[str, Any]:
    """
    Extracts key information from the summary using regex based on report type.

    Args:
        summary (str): The text summary generated by the model.
        report_type (str): Type of the report ('pathology_reports' or 'consultation_notes').

    Returns:
        Dict[str, Any]: Extracted key information with missing fields marked as 'was not inferred'.
    """
    extracted = {}
    if report_type == "pathology_reports":
        fields = [
            "Diagnosis",
            "Grade",
            "Tumor Size",
            "Clinical TNM Staging",
            "Pathological TNM Staging",
            "Lymph Node Status",
            "Resection Margins",
            "Biomarkers",
            "Others"
        ]
    elif report_type == "consultation_notes":
        fields = [
            "Yes/No/Cannot Infer",
            "Grade",
            "Tumor Size",
            "SUV from PET scans",
            "Pack Years",
            "Patient Concerns",
            "Recommendations",
            "Follow-up Actions",
            "Alcohol Consumption",
            "HPV Status",
            "Charlson Comorbidity Score",
            "ECOG Performance Status",
            "Karnofsky Performance Status",
            "Others"
        ]
    else:
        logger.warning(f"Unknown report type: {report_type}")
        return extracted

    for key in fields:
        pattern = PATTERNS.get(key)
        if pattern:
            match = re.search(pattern, summary, re.IGNORECASE | re.DOTALL)
            if match:
                if key == "Lymph Node Status":
                    extracted[key] = {
                        "Presence": match.group(1).strip() if match.group(1) else "was not inferred",
                        "Number of Lymph Nodes": match.group(2).strip() if match.group(2) else "was not inferred",
                        "Extranodal Extension": match.group(3).strip() if match.group(3) else "was not inferred"
                    }
                elif key == "Others":
                    extracted[key] = match.group(1).strip() if match.group(1) else "was not inferred"
                else:
                    extracted[key] = match.group(1).strip()
            else:
                extracted[key] = "was not inferred"
    return extracted

def validate_extracted_data(data: Dict[str, Any], report_type: str) -> bool:
    """
    Validates the extracted tabular data.

    Args:
        data (Dict[str, Any]): Extracted data for a single report.
        report_type (str): Type of the report ('pathology_reports' or 'consultation_notes').

    Returns:
        bool: True if data is valid, False otherwise.
    """
    if report_type == "pathology_reports":
        required_fields = [
            "Diagnosis",
            "Grade",
            "Tumor Size",
            "Clinical TNM Staging",
            "Pathological TNM Staging",
            "Lymph Node Status",
            "Resection Margins",
            "Biomarkers"
        ]
        # Validation rules
        tumor_size_pattern = r"^\d+(\.\d+)?$"  # Tumor Size in mm as a number
        grade_pattern = r"^[1-3]$"  # Assuming grades 1-3
        tnm_pattern = r"^[T]\d+[N]\d+[M]\d+$"  # Simplistic TNM staging format
    elif report_type == "consultation_notes":
        required_fields = [
            "Yes/No/Cannot Infer",
            "Grade",
            "Tumor Size",
            "SUV from PET scans",
            "Pack Years",
            "Patient Concerns",
            "Recommendations",
            "Follow-up Actions",
            "Alcohol Consumption",
            "HPV Status",
            "Charlson Comorbidity Score",
            "ECOG Performance Status",
            "Karnofsky Performance Status",
            "Others"
        ]
        # Validation rules
        tumor_size_pattern = r"^\d+(\.\d+)?$"  # Tumor Size in mm as a number
        grade_pattern = r"^[1-3]$"  # Assuming grades 1-3
    else:
        logger.warning(f"Unknown report type: {report_type}")
        return False

    # Check for presence of all required fields
    for field in required_fields:
        if field not in data or data[field] in [None, ""]:
            logger.debug(f"Missing or empty field '{field}' in data: {data}")
            return False

    # Specific validations
    if report_type == "pathology_reports":
        # Validate Tumor Size as numerical
        if not re.match(tumor_size_pattern, data["Tumor Size"]):
            logger.debug(f"Invalid Tumor Size value: {data['Tumor Size']}")
            return False

        # Validate Grade
        if not re.match(grade_pattern, data["Grade"]):
            logger.debug(f"Invalid Grade value: {data['Grade']}")
            return False

        # Validate TNM Staging formats
        if not re.match(tnm_pattern, data["Clinical TNM Staging"]):
            logger.debug(f"Invalid Clinical TNM Staging format: {data['Clinical TNM Staging']}")
            return False
        if not re.match(tnm_pattern, data["Pathological TNM Staging"]):
            logger.debug(f"Invalid Pathological TNM Staging format: {data['Pathological TNM Staging']}")
            return False

        # Validate Lymph Node Status
        lymph_status = data.get("Lymph Node Status", {})
        if not isinstance(lymph_status, dict):
            logger.debug(f"Lymph Node Status should be a dictionary: {lymph_status}")
            return False
        if lymph_status.get("Presence") not in ["Presence", "Absence", "was not inferred"]:
            logger.debug(f"Invalid Lymph Node Presence value: {lymph_status.get('Presence')}")
            return False
        if lymph_status.get("Extranodal Extension") not in ["Yes", "No", "was not inferred"]:
            logger.debug(f"Invalid Extranodal Extension value: {lymph_status.get('Extranodal Extension')}")
            return False
        if lymph_status.get("Number of Lymph Nodes") != "was not inferred":
            try:
                int(lymph_status.get("Number of Lymph Nodes"))
            except (ValueError, TypeError):
                logger.debug(f"Invalid Number of Lymph Nodes value: {data['Lymph Node Status']['Number of Lymph Nodes']}")
                return False

    elif report_type == "consultation_notes":
        # Validate Numerical Fields
        numerical_fields = ["Tumor Size", "SUV from PET scans", "Pack Years"]
        for field in numerical_fields:
            if data[field] != "was not inferred":
                try:
                    float(data[field]) if field != "Pack Years" else int(data[field])
                except (ValueError, TypeError):
                    logger.debug(f"Invalid {field} value: {data[field]}")
                    return False

        # Validate Grade
        if not re.match(grade_pattern, data["Grade"]):
            logger.debug(f"Invalid Grade value: {data['Grade']}")
            return False

        # Validate Categorical Data
        if data["Yes/No/Cannot Infer"] not in ["Yes", "No", "Cannot Infer"]:
            logger.debug(f"Invalid Categorical Data value: {data['Yes/No/Cannot Infer']}")
            return False

        # Validate Alcohol Consumption
        valid_alcohol = ["Never drank", "Ex-drinker", "Drinker", "Missing"]
        if data["Alcohol Consumption"] not in valid_alcohol:
            logger.debug(f"Invalid Alcohol Consumption value: {data['Alcohol Consumption']}")
            return False

        # Validate HPV Status
        valid_hpv = ["Positive", "Negative", "Missing"]
        if data["HPV Status"] not in valid_hpv:
            logger.debug(f"Invalid HPV Status value: {data['HPV Status']}")
            return False

        # Validate Performance & Comorbidity Scores
        try:
            if data["ECOG Performance Status"] != "Missing":
                if not (0 <= int(data["ECOG Performance Status"]) <= 5):
                    logger.debug(f"Invalid ECOG Performance Status value: {data['ECOG Performance Status']}")
                    return False
        except (ValueError, TypeError):
            logger.debug(f"Invalid ECOG Performance Status value: {data['ECOG Performance Status']}")
            return False

        try:
            if data["Karnofsky Performance Status"] != "Missing":
                if not (0 <= int(data["Karnofsky Performance Status"]) <= 100):
                    logger.debug(f"Invalid Karnofsky Performance Status value: {data['Karnofsky Performance Status']}")
                    return False
        except (ValueError, TypeError):
            logger.debug(f"Invalid Karnofsky Performance Status value: {data['Karnofsky Performance Status']}")
            return False

    return True

def normalize_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    """
    Normalizes and cleans the extracted tabular data.

    Args:
        df (pd.DataFrame): Raw extracted data.
        report_type (str): Type of the report ('pathology_reports' or 'consultation_notes').

    Returns:
        pd.DataFrame: Cleaned and normalized data.
    """
    if report_type == "pathology_reports":
        # Convert Tumor Size to float (already in mm)
        df['Tumor Size (mm)'] = pd.to_numeric(df['Tumor Size'], errors='coerce')
        df['Tumor Size (mm)'] = df['Tumor Size (mm)'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Convert Grades to integer
        df['Grade'] = pd.to_numeric(df['Grade'], errors='coerce').astype('Int64')
        df['Grade'] = df['Grade'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Normalize Clinical TNM Staging
        df['Clinical TNM Staging'] = df['Clinical TNM Staging'].str.upper().str.replace(" ", "").fillna("was not inferred")

        # Normalize Pathological TNM Staging
        df['Pathological TNM Staging'] = df['Pathological TNM Staging'].str.upper().str.replace(" ", "").fillna("was not inferred")

        # Normalize Lymph Node Status
        def normalize_lymph_status(status_dict):
            if not isinstance(status_dict, dict):
                return {"Presence": "was not inferred", "Number of Lymph Nodes": "was not inferred", "Extranodal Extension": "was not inferred"}
            return {
                "Presence": status_dict.get("Presence", "was not inferred"),
                "Number of Lymph Nodes": int(status_dict.get("Number of Lymph Nodes")) if isinstance(status_dict.get("Number of Lymph Nodes"), str) and status_dict.get("Number of Lymph Nodes") != "was not inferred" else status_dict.get("Number of Lymph Nodes"),
                "Extranodal Extension": status_dict.get("Extranodal Extension", "was not inferred")
            }

        df['Lymph Node Status'] = df['Lymph Node Status'].apply(normalize_lymph_status)

        # Normalize Resection Margins
        df['Resection Margins'] = df['Resection Margins'].str.capitalize().fillna("was not inferred")

        # Handle missing Biomarkers
        df['Biomarkers'] = df['Biomarkers'].fillna("was not inferred")

        # Handle "Others" if present
        if 'Others' in df.columns:
            df['Others'] = df['Others'].fillna("was not inferred")

    elif report_type == "consultation_notes":
        # Convert Grades to integer
        df['Grade'] = pd.to_numeric(df['Grade'], errors='coerce').astype('Int64')
        df['Grade'] = df['Grade'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Convert Tumor Size to float
        df['Tumor Size'] = pd.to_numeric(df['Tumor Size'], errors='coerce')
        df['Tumor Size'] = df['Tumor Size'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Convert SUV from PET scans to float
        df['SUV from PET scans'] = pd.to_numeric(df['SUV from PET scans'], errors='coerce')
        df['SUV from PET scans'] = df['SUV from PET scans'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Convert Pack Years to integer
        df['Pack Years'] = pd.to_numeric(df['Pack Years'], errors='coerce').astype('Int64')
        df['Pack Years'] = df['Pack Years'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Normalize Categorical Data
        df['Yes/No/Cannot Infer'] = df['Yes/No/Cannot Infer'].str.capitalize().fillna("was not inferred")

        # Normalize Lifestyle Factors
        df['Smoking History'] = df['Pack Years'].apply(
            lambda x: 'Never Smoked' if x == 0 else ('More Than 10 Years' if isinstance(x, int) and x > 10 else ('10 Years Smoking, 1 Pack/Day' if isinstance(x, int) and 0 < x <= 10 else "was not inferred"))
        )
        df['Alcohol Consumption'] = df['Alcohol Consumption'].str.capitalize().fillna("was not inferred")

        # Normalize HPV Status
        df['HPV Status'] = df['HPV Status'].str.capitalize().fillna("was not inferred")

        # Normalize Performance & Comorbidity Scores
        df['Charlson Comorbidity Score'] = pd.to_numeric(df['Charlson Comorbidity Score'], errors='coerce').astype('Int64')
        df['Charlson Comorbidity Score'] = df['Charlson Comorbidity Score'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        df['ECOG Performance Status'] = pd.to_numeric(df['ECOG Performance Status'], errors='coerce').astype('Int64')
        df['ECOG Performance Status'] = df['ECOG Performance Status'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        df['Karnofsky Performance Status'] = pd.to_numeric(df['Karnofsky Performance Status'], errors='coerce').astype('Int64')
        df['Karnofsky Performance Status'] = df['Karnofsky Performance Status'].apply(lambda x: x if not pd.isna(x) else "was not inferred")

        # Handle "Others" if present
        if 'Others' in df.columns:
            df['Others'] = df['Others'].fillna("was not inferred")

    return df

class ReportSummarizer:
    """
    Summarizes pathology reports and consultation notes, performs quality control,
    generates embeddings, converts summaries into structured tabular data, and encodes
    the structured data for survival analysis models.
    """

    def __init__(self, prompts_dir: str, model_type: str = "local", temperature: float = 0.3):
        """
        Initializes the summarizer with the specified model and prompts.

        Args:
            prompts_dir (str): Path to the directory containing prompt JSON files.
            model_type (str): Type of model to use ('local', 'gpt', 'gemini').
            temperature (float): Sampling temperature for the model.
        """
        self.model_type = model_type.lower()
        self.temperature = temperature

        # Load prompts from separate files based on report type
        # Assuming prompts_dir contains prompt_<report_type>.json files
        if not os.path.isdir(prompts_dir):
            logger.error(f"Prompts directory path {prompts_dir} is not a directory.")
            raise ValueError(f"Prompts directory path {prompts_dir} is not a directory.")

        self.prompts = {}
        for report_type in ["pathology_reports", "consultation_notes"]:
            prompt_file = os.path.join(prompts_dir, f"prompt_{report_type}.json")
            if os.path.isfile(prompt_file):
                with open(prompt_file, 'r') as f:
                    data = json.load(f)
                    self.prompts[report_type] = data.get("prompts", [])
                logger.info(f"Loaded {len(self.prompts[report_type])} prompts for {report_type}.")
            else:
                logger.warning(f"Prompt file {prompt_file} not found for report type {report_type}.")
                self.prompts[report_type] = []


        # Initialize the appropriate model and embeddings
        if self.model_type == "local":
            # self.model = ChatOllama(model="llama3.2:latest", temperature=self.temperature)
            self.model = ChatOllama(model="llama3.3:latest", temperature=self.temperature)
            logger.info(f"Initialized Llama 3.2 model.")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        elif self.model_type == "gpt":
            self.model = ChatOpenAI(model="gpt-4", temperature=self.temperature)
            self.embeddings = OpenAIEmbeddings()
        elif self.model_type == "gemini":
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=self.temperature)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Initialized embedding function using {model_type} embeddings.")

        # Initialize Prompt Templates using RunnableSequence
        self.chain_map = {}
        for report_type, prompt_list in self.prompts.items():
            if not prompt_list:
                continue
            # Select a random prompt from the list
            selected_prompt = random.choice(prompt_list)
            prompt = ChatPromptTemplate.from_template(selected_prompt)

            # Define a RunnableLambda for summarization
            def llm_runnable(inputs):
                context = inputs["context"]
                try:
                    # Pass the context as a single string with an instruction
                    prompt_text = f"Please summarize the following report:\n\n{context}"
                    # response = self.model.generate(prompt_text)
                    response = self.model.invoke([HumanMessage(content=prompt_text)]).content
                    return {"summary": response}
                except Exception as e:
                    logger.error(f"Error in LLM generation: {e}")
                    return {"summary": ""}

            llm_lambda = RunnableLambda(llm_runnable)

            # Create a RunnableSequence with the summarization runnable
            # sequence = RunnableSequence(runnables=[llm_lambda])
            sequence = llm_lambda

            self.chain_map[report_type] = sequence

        logger.info(f"Initialized LLM chains for report types: {list(self.chain_map.keys())}")
        # logger.info(f"Initialized LLM chains for report types: {list(self.chain_map.keys())[0]}")

    def summarize_report(self, report_text: str, report_type: str) -> Optional[str]:
        """
        Generates a summary for a single report.

        Args:
            report_text (str): The raw text of the report.
            report_type (str): Type of the report ('pathology_reports' or 'consultation_notes').

        Returns:
            Optional[str]: The generated summary or None if generation failed.
        """
        try:
            if report_type not in self.chain_map:
                logger.warning(f"No prompts available for report type: {report_type}")
                return None

            chain = self.chain_map[report_type]
            summary_output = chain.invoke({"context": report_text})
            summary = summary_output.get("summary", "").strip()
            return summary if summary else None
        except Exception as e:
            logger.error(f"Error generating summary for report type {report_type}: {e}")
            return None

    def encode_structured_data(self, df: pd.DataFrame, report_type: str) -> pd.DataFrame:
        """
        Encodes structured data for survival analysis models.

        Args:
            df (pd.DataFrame): Structured data DataFrame.
            report_type (str): Type of the report ('pathology_reports' or 'consultation_notes').

        Returns:
            pd.DataFrame: Encoded DataFrame ready for survival analysis.
        """
        df_encoded = df.copy()

        # Define categorical and numerical columns based on report type
        if report_type == "pathology_reports":
            categorical_cols = [
                "Diagnosis",
                "Clinical TNM Staging",
                "Pathological TNM Staging",
                "Lymph Node Status.Presence",
                "Lymph Node Status.Extranodal Extension",
                "Resection Margins",
                "Biomarkers",
                "Others"
            ]
            numerical_cols = [
                "Grade",
                "Tumor Size (mm)",
                "Lymph Node Status.Number of Lymph Nodes"
            ]
        elif report_type == "consultation_notes":
            categorical_cols = [
                "Yes/No/Cannot Infer",
                "Smoking History",
                "Alcohol Consumption",
                "HPV Status",
                "Others"
            ]
            numerical_cols = [
                "Grade",
                "Tumor Size",
                "SUV from PET scans",
                "Pack Years",
                "Charlson Comorbidity Score",
                "ECOG Performance Status",
                "Karnofsky Performance Status"
            ]
        else:
            logger.warning(f"Unknown report type for encoding: {report_type}")
            return df_encoded

        # Handle missing numerical data
        numerical_imputer = SimpleImputer(strategy='median')
        df_encoded[numerical_cols] = numerical_imputer.fit_transform(df_encoded[numerical_cols])

        # Replace 'was not inferred' with a placeholder for encoding
        df_encoded.replace("was not inferred", "Missing", inplace=True)

        # Encode Categorical Variables using One-Hot Encoding
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(df_encoded[categorical_cols])

        # Create DataFrame for Encoded Categorical Variables
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        # Combine Numerical and Encoded Categorical Data
        df_final = pd.concat([df_encoded[numerical_cols].reset_index(drop=True), encoded_categorical_df], axis=1)

        return df_final

    def process_reports(self, input_dir: str, output_dir: str) -> None:
        """
        Processes all .txt reports in the input directory and its subdirectories.

        Args:
            input_dir (str): Directory containing subdirectories for report types.
            output_dir (str): Directory to save summaries, embeddings, and tabular data.
        """
        summaries = []
        tabular_data = []
        invalid_entries = []

        # Recursively traverse the input directory to find all .txt files
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            report_text = f.read()

                        # Determine report_type based on the parent directory name
                        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
                        if parent_dir == "pathology_reports":
                            report_type = "pathology_reports"
                        elif parent_dir == "consultation_notes":
                            report_type = "consultation_notes"
                        else:
                            logger.warning(f"Unknown report type for file: {filename}. Skipping.")
                            invalid_entries.append({"file": filename, "reason": "Unknown report type"})
                            continue

                        # Generate summary
                        summary = self.summarize_report(report_text, report_type)
                        if not summary:
                            logger.warning(f"Failed to generate summary for {filename}.")
                            invalid_entries.append({"file": filename, "reason": "Summary generation failed"})
                            continue

                        # Extract tabular data
                        logger.info(f"Extracting tabular data for {filename} ({report_type})")
                        extracted = extract_tabular_data(summary, report_type)

                        # Validate extracted data
                        if validate_extracted_data(extracted, report_type):
                            summaries.append({"file": filename, "report_type": report_type, "summary": summary})
                            tabular_data.append({"file": filename, "report_type": report_type, "data": extracted})
                        else:
                            logger.warning(f"Validation failed for {filename}. Summary may be incomplete.")
                            invalid_entries.append({"file": filename, "reason": "Validation failed"})
                            summaries.append({"file": filename, "report_type": report_type, "summary": summary})  # Still save the summary
                            tabular_data.append({"file": filename, "report_type": report_type, "data": extracted})    # Save whatever was extracted

                        # Determine base filename without extension
                        base_filename = os.path.splitext(filename)[0]

                        # Create subfolder path based on base_filename
                        summary_dir = os.path.join(output_dir, "text_summaries", base_filename)
                        os.makedirs(summary_dir, exist_ok=True)
                        summary_filename = f"{report_type}_summary.txt"
                        summary_path = os.path.join(summary_dir, summary_filename)
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            f.write(summary)
                        logger.info(f"Summary saved to {summary_path}")

                        # Generate and Save Embeddings
                        embedding_dir = os.path.join(output_dir, "embeddings", base_filename)
                        os.makedirs(embedding_dir, exist_ok=True)
                        embedding_filename = f"{report_type}_embedding.pkl"
                        embedding_path = os.path.join(embedding_dir, embedding_filename)
                        try:
                            # Directly use the embeddings object to generate embeddings
                            embeddings_generated = self.embeddings.embed_documents([summary])[0]
                            with open(embedding_path, 'wb') as f:
                                pickle.dump(embeddings_generated, f)
                            logger.info(f"Embedding saved to {embedding_path}")
                        except Exception as e:
                            logger.error(f"Error generating/saving embeddings for {filename}: {e}")
                            invalid_entries.append({"file": filename, "reason": "Embedding generation failed"})

                        # Save Structured Data
                        structured_dir = os.path.join(output_dir, "structured_data", base_filename)
                        os.makedirs(structured_dir, exist_ok=True)
                        structured_filename = f"{report_type}_structured.csv"
                        structured_path = os.path.join(structured_dir, structured_filename)
                        structured_df = pd.DataFrame([extracted])
                        structured_df.to_csv(structured_path, index=False)
                        logger.info(f"Structured data saved to {structured_path}")

                        # Encode and Save Structured Data for Survival Analysis
                        encoded_dir = os.path.join(output_dir, "structured_data_encoded", base_filename)
                        os.makedirs(encoded_dir, exist_ok=True)
                        encoded_filename = f"{report_type}_structured_encoded.csv"
                        encoded_path = os.path.join(encoded_dir, encoded_filename)
                        try:
                            encoded_df = self.encode_structured_data(structured_df, report_type)
                            encoded_df.to_csv(encoded_path, index=False)
                            logger.info(f"Encoded structured data saved to {encoded_path}")
                        except Exception as e:
                            logger.error(f"Error encoding structured data for {filename}: {e}")
                            invalid_entries.append({"file": filename, "reason": "Structured data encoding failed"})

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        invalid_entries.append({"file": filename, "reason": str(e)})

        # Save Summaries Metadata (Optional)
        if summaries:
            summaries_df = pd.DataFrame(summaries)
            summaries_csv = os.path.join(output_dir, "summaries_metadata.csv")
            summaries_df.to_csv(summaries_csv, index=False)
            logger.info(f"Summaries metadata saved to {summaries_csv}")

        # Save Tabular Data Metadata (Optional)
        if tabular_data:
            tabular_df = pd.DataFrame(tabular_data)
            tabular_csv = os.path.join(output_dir, "tabular_data_metadata.csv")
            tabular_df.to_csv(tabular_csv, index=False)
            logger.info(f"Tabular data metadata saved to {tabular_csv}")

        # Save Invalid Entries Log
        if invalid_entries:
            invalid_df = pd.DataFrame(invalid_entries)
            invalid_log = os.path.join(output_dir, "invalid_entries.log")
            invalid_df.to_csv(invalid_log, index=False)
            logger.info(f"Invalid entries logged to {invalid_log}")

def main():
    """
    Main function to execute the summarization pipeline.
    """
    parser = argparse.ArgumentParser(description="Pathology Report and Consultation Notes Summarizer using Llama 3.3 and LangChain")

    parser.add_argument(
        '--prompts_dir',
        type=str,
        required=True,
        help='Directory containing separate JSON prompt files for each report type.'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='local',
        choices=['local', 'gpt', 'gemini'],
        help='Type of model to use for summarization: local (Llama 3.3 via Ollama), gpt (OpenAI GPT), gemini (Google Generative AI)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature for the language model.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing subdirectories for consultation_notes and pathology_reports with .txt files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save summaries, embeddings, and tabular data.'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='ollama',
        choices=['openai', 'ollama', 'google'],
        help='Type of embedding model to use for generating embeddings: openai, ollama, google'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Summarizer
    summarizer = ReportSummarizer(prompts_dir=args.prompts_dir, model_type=args.model_type, temperature=args.temperature)

    # Process Reports
    summarizer.process_reports(input_dir=args.input_dir, output_dir=args.output_dir)

    # Note: Encoding is handled within process_reports method

if __name__ == "__main__":
    main()
