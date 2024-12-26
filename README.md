# HNC Clinical Reports Agents for Key Information Extraction

## Project Overview
This project focuses on transforming unstructured clinical reports (e.g., pathology reports, consultation notes) into structured formats. This involves extracting key information, summarizing texts, and generating structured variables and embeddings for downstream tasks like survival analysis.

---

## Repository Structure
```
/Data/Yujing/HNC_OutcomePred/Reports_Agents/
├── input_reports/
│   ├── consultation_notes/
│   │   ├── consultation_notes_example1.txt
│   │   ├── consultation_notes_example2.txt
│   │   └── subfolder/
│   │       └── consultation_notes_example3.txt
│   └── pathology_reports/
│       ├── pathology_report_example1.txt
│       ├── pathology_report_example2.txt
│       └── subfolder/
│           └── pathology_report_example3.txt
├── prompts/
│   ├── prompt_consultation_notes.json
│   └── prompt_pathology_report.json
├── output_dir/
│   ├── text_summaries/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_summary.txt
│   │   │   └── patient_id2/
│   │   │       └── consultation_notes_summary.txt
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_summary.txt
│   │       └── patient_id2/
│   │           └── pathology_reports_summary.txt
│   ├── embeddings/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_embedding.pkl
│   │   │   └── patient_id2/
│   │   │       └── consultation_notes_embedding.pkl
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_embedding.pkl
│   │       └── patient_id2/
│   │           └── pathology_reports_embedding.pkl
│   ├── structured_data/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_structured.csv
│   │   │   └── patient_id2/
│   │   │       └── consultation_notes_structured.csv
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_structured.csv
│   │       └── patient_id2/
│   │           └── pathology_reports_structured.csv
│   ├── structured_data_encoded/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_structured_encoded.csv
│   │   │   └── patient_id2/
│   │   │       └── consultation_notes_structured_encoded.csv
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_structured_encoded.csv
│   │       └── patient_id2/
│   │           └── pathology_reports_structured_encoded.csv
├── summaries_metadata.csv
├── tabular_data_metadata.csv
└── invalid_entries.log
```

---

## Workflow
### 1. **Document Ingestion**
- Collect `.txt` files containing pathology reports and consultation notes from the `input_reports/` directory.

### 2. **Prompt Configuration/Engineering**
- Define prompts in separate JSON files under `prompts/` (e.g., `prompt_consultation_notes.json`, `prompt_pathology_report.json`).

### 3. **Model Initialization**
- Initialize a language model (e.g., Llama 3.3, GPT, Google Gemini) based on user selection. Configure parameters like temperature for response variability.

### 4. **Summary Generation**
- Generate concise summaries of the reports using LangChain with the defined prompts.

### 5. **Quality Control**
- Automate validation checks to ensure the summaries are complete and meet format requirements. Log discrepancies in `invalid_entries.log`.

### 6. **Post-Processing**
- Extract structured variables from summaries using tailored regular expressions.

### 7. **Normalization**
- Standardize data (e.g., convert tumor sizes to millimeters, handle missing values).

### 8. **Embedding Generation**
- Generate embeddings using specified models for downstream tasks like clustering and similarity searches.

### 9. **Output Storage**
- Store results (summaries, structured data, embeddings) in organized directories under `output_dir/`.

### 10. **Logging and Error Handling**
- Maintain logs of processing statuses and errors for manual review.

---

## Installation
### Repository Setup
```bash
cd /Data/Yujing/HNC_OutcomePred/Reports_Agents
git init
git remote add origin https://github.com/yujing1997/HNC_Reports_Agents.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

### Virtual Environment Setup
```bash
python3 -m venv env
source env/bin/activate
```
OR
```bash
conda activate segment2
conda install langchain chromadb openai google-generative-ai tqdm pandas
```
Add other necessary installations based on your models and embeddings.

### Package Requirements Quirks
```bash
conda install pip
pip install --upgrade pip
pip install -U langchain-ollama
pip show langchain-ollama
pip install -qU langchain-openai
```

---

## Notes on Reports and Prompt Engineering
### Pathology Reports
- Structure extracted variables such as:
  - Numerical metrics (e.g., tumor size).
  - Patient history.
  - Anatomic site of lesion.
  - Lymph node status (presence/absence, count).
  - HPV or p16 status.
  - Smoking and alcohol history.
  - Immunohistochemistry results.

### Privacy-Preserving Workflow
- Use local LLMs (e.g., Llama 3.3) on privacy-compliant infrastructure.

---

## Future Work
1. **Extremely Early Fusion**
   - Combine clinical variable tabular data with reports before embedding generation.

2. **Integration with Multimodal Systems**
   - Extend the pipeline to include CT/PET images, dosimetry RT files, and histopathology slides.

3. **RAG and CoT Enhancements**
   - Evaluate Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) approaches for clinical predictions.

4. **External Validation**
   - Validate using datasets like TCIA, GDC, and other public datasets.

---

## Contributions
- Prompt engineering: Yujing, Dr. Khalil Sultanem, Dr. George Shenouda.
- Code adaptation: Yujing, Laya, Farhood.

---

## References
- Expert-guided knowledge extraction (JCO Precision Oncology).
- LLM-for-Survival-Analysis (https://github.com/yujing1997/LLM-For-Survival-Analysis).
- RAG inspiration: https://github.com/engerlab/medphys-agent.

---

## Issues and Improvements
### Output Directory
- Ensure uniform output structures for summaries and embeddings.
- Resolve issues in `structured_data` and `structured_data_encoded`.

### Prompt Refinement
- Tailor prompts for more succinct and structured outputs.

---
