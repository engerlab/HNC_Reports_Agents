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
from langchain_core.runnables import RunnableLambda
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
    # Pathology Reports Structured Prompt Patterns
    "Quantitative_Numerical_Metrics": r"Quantitative/Numerical Metrics:\s*(.*)",
    "Patient_History_and_Status": r"Patient_History_and_Status:\s*(.*)",
    "Anatomic_Site_of_Lesion": r"Anatomic_Site_of_Lesion:\s*(.*)",
    "Lymph_Node_Status_Presence_Absence": r"Presence_Absence:\s*(Presence|Absence|Not inferred)",
    "Lymph_Node_Status_Number_of_Lymph_Nodes": r"Number_of_Lymph_Nodes:\s*(\d+|Not inferred)",
    "Lymph_Node_Status_Extranodal_Extension": r"Extranodal_Extension:\s*(Yes|No|Not inferred)",
    "Resection_Margin": r"Resection_Margin:\s*(Positive|Negative|Not inferred)",
    "HPV_or_p16_Status": r"HPV_or_p16_Status:\s*(Positive|Negative|Not inferred)",
    "Smoking_History": r"Smoking_History:\s*(Never smoked|10 years smoking 1 pack a day|More than 10 years|Not inferred)",
    "Alcohol_Consumption": r"Alcohol_Consumption:\s*(Never drank|Ex-drinker|Drinker|Not inferred)",
    "Specific_Immunohistochemistry_Biomarkers": r"Specific_Immunohistochemistry_Biomarkers:\s*([A-Za-z0-9\+\(\),\s\-]+|Not inferred)",
    "Others": r"Others:\s*(.*)",

    # Consultation Notes Structured Prompt Patterns
    "Yes_No_Cannot_Infer": r"Yes_No_Cannot_Infer:\s*(Yes|No|Cannot Infer)",
    "Grade": r"Grade:\s*([1-3]|Not inferred)",
    "Tumor_Size": r"Tumor_Size:\s*(\d+(\.\d+)?|Not inferred)",
    "SUV_from_PET_scans": r"SUV_from_PET_scans:\s*(\d+(\.\d+)?|Not inferred)",
    "Pack_Years": r"Pack_Years:\s*(\d+|Not inferred)",
    "Patient_Concerns": r"Patient_Concerns:\s*(.*)",
    "Recommendations": r"Recommendations:\s*(.*)",
    "Follow_up_Actions": r"Follow_up_Actions:\s*(.*)",
    "Alcohol_Consumption_Consult": r"Alcohol_Consumption:\s*(Never drank|Ex-drinker|Drinker|Not inferred)",
    "HPV_Status": r"HPV_Status:\s*(Positive|Negative|Not inferred)",
    "Charlson_Comorbidity_Score": r"Charlson_Comorbidity_Score:\s*(\d+|Not inferred)",
    "ECOG_Performance_Status": r"ECOG_Performance_Status:\s*(0|1|4|Not inferred)",
    "Karnofsky_Performance_Status": r"Karnofsky_Performance_Status:\s*(100|0|Not inferred)",
    "Patient_History_Status_Former_Status_of_Patient": r"Patient_History_Status_Former_Status_of_Patient:\s*(.*)",
    "Patient_History_Status_Similar_Conditions": r"Patient_History_Status_Similar_Conditions:\s*(.*)",
    "Patient_History_Status_Previous_Treatments": r"Patient_History_Status_Previous_Treatments:\s*(Radiation|Chemotherapy|Surgery|None|Not inferred)",
    "Clinical_Assessments_Radiological_Lesions": r"Clinical_Assessments_Radiological_Lesions:\s*(.*)",
    "Clinical_Assessments_SUV_from_PET_Scans": r"Clinical_Assessments_SUV_from_PET_Scans:\s*(\d+(\.\d+)?|Not inferred)",
    "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score": r"Performance_Comorbidity_Scores_Charlson_Comorbidity_Score:\s*(\d+|Not inferred)",
    "Performance_Comorbidity_Scores_ECOG_Performance_Status": r"Performance_Comorbidity_Scores_ECOG_Performance_Status:\s*(0|1|4|Not inferred)",
    "Performance_Comorbidity_Scores_Karnofsky_Performance_Status": r"Performance_Comorbidity_Scores_Karnofsky_Performance_Status:\s*(100|0|Not inferred)",
    "Cancer_Staging_Pathological_TNM": r"Cancer_Staging_Pathological_TNM:\s*([T]\d+[N]\d+[M]\d+|Not inferred)",
    "Cancer_Staging_Clinical_TNM": r"Cancer_Staging_Clinical_TNM:\s*([T]\d+[N]\d+[M]\d+|Not inferred)",
    "Cancer_Staging_Tumor_Size": r"Cancer_Staging_Tumor_Size:\s*(\d+(\.\d+)?|Not inferred)"
}

def extract_tabular_data(summary: str, report_type: str) -> Dict[str, Any]:
    """
    Extracts key information from the summary using regex based on report type.

    Args:
        summary (str): The text summary generated by the model.
        report_type (str): Type of the report ('pathology_reports' or 'consultation_notes').

    Returns:
        Dict[str, Any]: Extracted key information with missing fields marked as 'Not inferred'.
    """
    extracted = {}
    if report_type == "pathology_reports":
        fields = [
            "Quantitative_Numerical_Metrics",
            "Patient_History_and_Status",
            "Anatomic_Site_of_Lesion",
            "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Number_of_Lymph_Nodes",
            "Lymph_Node_Status_Extranodal_Extension",
            "Resection_Margin",
            "HPV_or_p16_Status",
            "Smoking_History",
            "Alcohol_Consumption",
            "Specific_Immunohistochemistry_Biomarkers",
            "Others"
        ]
    elif report_type == "consultation_notes":
        fields = [
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
            "Clinical_Assessments_SUV_from_PET_Scans",
            "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score",
            "Performance_Comorbidity_Scores_ECOG_Performance_Status",
            "Performance_Comorbidity_Scores_Karnofsky_Performance_Status",
            "Cancer_Staging_Pathological_TNM",
            "Cancer_Staging_Clinical_TNM",
            "Cancer_Staging_Tumor_Size",
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
                extracted[key] = match.group(1).strip()
            else:
                extracted[key] = "Not inferred"
                logger.debug(f"Pattern not matched for key '{key}' in summary: {summary}")
        else:
            logger.debug(f"No pattern defined for key '{key}'")
            extracted[key] = "Not inferred"
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
            "Quantitative_Numerical_Metrics",
            "Patient_History_and_Status",
            "Anatomic_Site_of_Lesion",
            "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Number_of_Lymph_Nodes",
            "Lymph_Node_Status_Extranodal_Extension",
            "Resection_Margin",
            "HPV_or_p16_Status",
            "Smoking_History",
            "Alcohol_Consumption",
            "Specific_Immunohistochemistry_Biomarkers"
        ]
        # Validation rules
        grade_pattern = r"^(Grade\s*\d+|Grade\s*[IVX]+)$"  # Supports both numerical and Roman numeral grades
    elif report_type == "consultation_notes":
        required_fields = [
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
            "Clinical_Assessments_SUV_from_PET_Scans",
            "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score",
            "Performance_Comorbidity_Scores_ECOG_Performance_Status",
            "Performance_Comorbidity_Scores_Karnofsky_Performance_Status",
            "Cancer_Staging_Pathological_TNM",
            "Cancer_Staging_Clinical_TNM",
            "Cancer_Staging_Tumor_Size"
        ]
        # Validation rules
        grade_pattern = r"^[1-3]$"  # Assuming grades 1-3
    else:
        logger.warning(f"Unknown report type: {report_type}")
        return False

    # Check for presence of all required fields
    for field in required_fields:
        if field not in data or not data[field]:
            logger.debug(f"Missing or empty field '{field}' in data: {data}")
            return False

    # Specific validations
    if report_type == "pathology_reports":
        # Example: Validate that "Quantitative_Numerical_Metrics" is a number or 'Not inferred'
        if data["Quantitative_Numerical_Metrics"] != "Not inferred":
            if not re.match(r"^\d+(\.\d+)?\s*(mm|cm|in)?$", data["Quantitative_Numerical_Metrics"]):
                logger.debug(f"Invalid Quantitative/Numerical Metrics value: {data['Quantitative_Numerical_Metrics']}")
                return False

        # Add more validations as needed
    elif report_type == "consultation_notes":
        # Example: Validate "Tumor_Size" is a number or 'Not inferred'
        if data["Tumor_Size"] != "Not inferred":
            try:
                float(data["Tumor_Size"])
            except ValueError:
                logger.debug(f"Invalid Tumor Size value: {data['Tumor_Size']}")
                return False

        # Validate Grade
        if not re.match(grade_pattern, data["Grade"]):
            logger.debug(f"Invalid Grade value: {data['Grade']}")
            return False

        # Validate SUV from PET scans
        if data["SUV_from_PET_scans"] != "Not inferred":
            try:
                float(data["SUV_from_PET_scans"])
            except ValueError:
                logger.debug(f"Invalid SUV from PET scans value: {data['SUV_from_PET_scans']}")
                return False

        # Validate ECOG Performance Status
        if data["ECOG_Performance_Status"] not in ["0", "1", "4", "Not inferred"]:
            logger.debug(f"Invalid ECOG Performance Status value: {data['ECOG_Performance_Status']}")
            return False

        # Validate Karnofsky Performance Status
        if data["Karnofsky_Performance_Status"] not in ["100", "0", "Not inferred"]:
            logger.debug(f"Invalid Karnofsky Performance Status value: {data['Karnofsky_Performance_Status']}")
            return False

        # Add more validations as needed

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
        # Convert Quantitative_Numerical_Metrics to float
        df['Quantitative_Numerical_Metrics'] = pd.to_numeric(df['Quantitative_Numerical_Metrics'], errors='coerce')
        # Do NOT replace NaN with "Not inferred"

        # Normalize Categorical Fields
        categorical_fields = [
            "Patient_History_and_Status",
            "Anatomic_Site_of_Lesion",
            "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Extranodal_Extension",
            "Resection_Margin",
            "HPV_or_p16_Status",
            "Smoking_History",
            "Alcohol_Consumption"
        ]
        for field in categorical_fields:
            df[field] = df[field].str.title().fillna("Missing")

        # Handle Specific Immunohistochemistry Biomarkers
        df['Specific_Immunohistochemistry_Biomarkers'] = df['Specific_Immunohistochemistry_Biomarkers'].fillna("Missing")

    elif report_type == "consultation_notes":
        # Convert numerical fields
        numerical_fields = ["Grade", "Tumor_Size", "SUV_from_PET_scans", "Pack_Years", "Charlson_Comorbidity_Score", "ECOG_Performance_Status", "Karnofsky_Performance_Status"]
        for field in numerical_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(pd.NA)

        # Normalize Categorical Fields
        categorical_fields = [
            "Yes_No_Cannot_Infer",
            "Alcohol_Consumption_Consult",
            "HPV_Status",
            "Patient_History_Status_Former_Status_of_Patient",
            "Patient_History_Status_Similar_Conditions",
            "Patient_History_Status_Previous_Treatments",
            "Clinical_Assessments_Radiological_Lesions",
            "Clinical_Assessments_SUV_from_PET_Scans",
            "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score",
            "Performance_Comorbidity_Scores_ECOG_Performance_Status",
            "Performance_Comorbidity_Scores_Karnofsky_Performance_Status",
            "Cancer_Staging_Pathological_TNM",
            "Cancer_Staging_Clinical_TNM",
            "Cancer_Staging_Tumor_Size",
            "Others"
        ]
        for field in categorical_fields:
            df[field] = df[field].str.title().fillna("Missing")

    return df

def encode_structured_data(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
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
            "Patient_History_and_Status",
            "Anatomic_Site_of_Lesion",
            "Lymph_Node_Status_Presence_Absence",
            "Lymph_Node_Status_Extranodal_Extension",
            "Resection_Margin",
            "HPV_or_p16_Status",
            "Smoking_History",
            "Alcohol_Consumption",
            "Specific_Immunohistochemistry_Biomarkers",
            "Others"
        ]
        numerical_cols = [
            "Quantitative_Numerical_Metrics"
        ]
    elif report_type == "consultation_notes":
        categorical_cols = [
            "Yes_No_Cannot_Infer",
            "Alcohol_Consumption_Consult",
            "HPV_Status",
            "Patient_History_Status_Former_Status_of_Patient",
            "Patient_History_Status_Similar_Conditions",
            "Patient_History_Status_Previous_Treatments",
            "Clinical_Assessments_Radiological_Lesions",
            "Clinical_Assessments_SUV_from_PET_Scans",
            "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score",
            "Performance_Comorbidity_Scores_ECOG_Performance_Status",
            "Performance_Comorbidity_Scores_Karnofsky_Performance_Status",
            "Cancer_Staging_Pathological_TNM",
            "Cancer_Staging_Clinical_TNM",
            "Cancer_Staging_Tumor_Size",
            "Others"
        ]
        numerical_cols = [
            "Grade",
            "Tumor_Size",
            "SUV_from_PET_scans",
            "Pack_Years",
            "Charlson_Comorbidity_Score",
            "ECOG_Performance_Status",
            "Karnofsky_Performance_Status"
        ]
    else:
        logger.warning(f"Unknown report type for encoding: {report_type}")
        return df_encoded

    # Handle missing numerical data
    numerical_imputer = SimpleImputer(strategy='median')
    if numerical_cols:
        # The numerical columns already have pd.NA where data is missing
        df_encoded[numerical_cols] = numerical_imputer.fit_transform(df_encoded[numerical_cols])

    # Encode Categorical Variables using One-Hot Encoding
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    if categorical_cols:
        encoded_categorical = encoder.fit_transform(df_encoded[categorical_cols])

        # Create DataFrame for Encoded Categorical Variables
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=encoder.get_feature_names_out(categorical_cols)
        )
    else:
        encoded_categorical_df = pd.DataFrame()

    # Combine Numerical and Encoded Categorical Data
    df_final = pd.concat([pd.DataFrame(df_encoded[numerical_cols], columns=numerical_cols).reset_index(drop=True), encoded_categorical_df], axis=1)

    return df_final

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
            # Update the model name based on available models in Ollama
            # Replace "llama-3.3" with the exact model name as listed by `ollama list`
            self.model = ChatOllama(model="llama3.3:latest", temperature=self.temperature)
            logger.info(f"Initialized Llama 3.3 model.")
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

        # Initialize Prompt Templates using RunnableLambda
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

            # Assign RunnableLambda directly to chain_map
            self.chain_map[report_type] = llm_lambda

        logger.info(f"Initialized LLM chains for report types: {list(self.chain_map.keys())}")

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

            runnable = self.chain_map[report_type]
            summary_output = runnable.invoke({"context": report_text})
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
                "Patient_History_and_Status",
                "Anatomic_Site_of_Lesion",
                "Lymph_Node_Status_Presence_Absence",
                "Lymph_Node_Status_Extranodal_Extension",
                "Resection_Margin",
                "HPV_or_p16_Status",
                "Smoking_History",
                "Alcohol_Consumption",
                "Specific_Immunohistochemistry_Biomarkers",
                "Others"
            ]
            numerical_cols = [
                "Quantitative_Numerical_Metrics"
            ]
        elif report_type == "consultation_notes":
            categorical_cols = [
                "Yes_No_Cannot_Infer",
                "Alcohol_Consumption_Consult",
                "HPV_Status",
                "Patient_History_Status_Former_Status_of_Patient",
                "Patient_History_Status_Similar_Conditions",
                "Patient_History_Status_Previous_Treatments",
                "Clinical_Assessments_Radiological_Lesions",
                "Clinical_Assessments_SUV_from_PET_Scans",
                "Performance_Comorbidity_Scores_Charlson_Comorbidity_Score",
                "Performance_Comorbidity_Scores_ECOG_Performance_Status",
                "Performance_Comorbidity_Scores_Karnofsky_Performance_Status",
                "Cancer_Staging_Pathological_TNM",
                "Cancer_Staging_Clinical_TNM",
                "Cancer_Staging_Tumor_Size",
                "Others"
            ]
            numerical_cols = [
                "Grade",
                "Tumor_Size",
                "SUV_from_PET_scans",
                "Pack_Years",
                "Charlson_Comorbidity_Score",
                "ECOG_Performance_Status",
                "Karnofsky_Performance_Status"
            ]
        else:
            logger.warning(f"Unknown report type for encoding: {report_type}")
            return df_encoded

        # Handle missing numerical data
        numerical_imputer = SimpleImputer(strategy='median')
        if numerical_cols:
            # The numerical columns already have pd.NA where data is missing
            df_encoded[numerical_cols] = numerical_imputer.fit_transform(df_encoded[numerical_cols])

        # Encode Categorical Variables using One-Hot Encoding
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        if categorical_cols:
            encoded_categorical = encoder.fit_transform(df_encoded[categorical_cols])

            # Create DataFrame for Encoded Categorical Variables
            encoded_categorical_df = pd.DataFrame(
                encoded_categorical,
                columns=encoder.get_feature_names_out(categorical_cols)
            )
        else:
            encoded_categorical_df = pd.DataFrame()

        # Combine Numerical and Encoded Categorical Data
        df_final = pd.concat([pd.DataFrame(df_encoded[numerical_cols], columns=numerical_cols).reset_index(drop=True), encoded_categorical_df], axis=1)

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

        # Define mapping from parent directory to report type
        report_type_mapping = {
            "pathology_reports": "pathology_reports",
            "consultation_notes": "consultation_notes"
        }

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
                        report_type = report_type_mapping.get(parent_dir)
                        if not report_type:
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

                        # Determine patient ID (assuming filename without extension is patient ID)
                        patient_id = os.path.splitext(filename)[0]

                        # Define paths for each output type
                        paths = {
                            "text_summaries": os.path.join(output_dir, "text_summaries", report_type, patient_id),
                            "embeddings": os.path.join(output_dir, "embeddings", report_type, patient_id),
                            "structured_data": os.path.join(output_dir, "structured_data", report_type, patient_id),
                            "structured_data_encoded": os.path.join(output_dir, "structured_data_encoded", report_type, patient_id)
                        }

                        # Create necessary directories
                        for path in paths.values():
                            os.makedirs(path, exist_ok=True)

                        # Save Summary
                        summary_filename = f"{report_type}_summary.txt"
                        summary_path = os.path.join(paths["text_summaries"], summary_filename)
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            f.write(summary)
                        logger.info(f"Summary saved to {summary_path}")

                        # Generate and Save Embeddings
                        embedding_filename = f"{report_type}_embedding.pkl"
                        embedding_path = os.path.join(paths["embeddings"], embedding_filename)
                        try:
                            # Generate embeddings for the summarized text
                            embeddings_generated = self.embeddings.embed_documents([summary])[0]
                            with open(embedding_path, 'wb') as f:
                                pickle.dump(embeddings_generated, f)
                            logger.info(f"Embedding saved to {embedding_path}")
                        except Exception as e:
                            logger.error(f"Error generating/saving embeddings for {filename}: {e}")
                            invalid_entries.append({"file": filename, "reason": "Embedding generation failed"})

                        # Save Structured Data
                        structured_filename = f"{report_type}_structured.csv"
                        structured_path = os.path.join(paths["structured_data"], structured_filename)
                        try:
                            structured_df = pd.DataFrame([extracted])
                            structured_df = normalize_data(structured_df, report_type)
                            structured_df.to_csv(structured_path, index=False)
                            logger.info(f"Structured data saved to {structured_path}")
                        except Exception as e:
                            logger.error(f"Error saving structured data for {filename}: {e}")
                            invalid_entries.append({"file": filename, "reason": "Structured data saving failed"})
                            continue  # Skip encoding if structured data saving failed

                        # Encode and Save Structured Data for Survival Analysis
                        encoded_filename = f"{report_type}_structured_encoded.csv"
                        encoded_path = os.path.join(paths["structured_data_encoded"], encoded_filename)
                        try:
                            encoded_df = encode_structured_data(structured_df, report_type)
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
