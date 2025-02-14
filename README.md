# HNC Clinical Reports Agents for Key Information Extraction

## Project Overview
This project focuses on transforming unstructured clinical reports (e.g., pathology reports, consultation notes) into structured formats. This involves extracting key information, summarizing texts, and generating structured variables and embeddings for downstream tasks like survival analysis.

<div align="center">
  <img src="https://github.com/engerlab/HNC_Reports_Agents/blob/1cabc6acb89f1466c943f30ab2955b5be4809e6a/img/Head%20%26%20Neck%20Clinician-AI%20Meetings.png" 
  alt="Alt text" width="75%">
</div>


---

## Scripts 
- `hnc_reports_agent2.py`: Expert-guided LLM extraction of Pathology & Consultation Notes, and Treatment Plan & Outcome Prediction medical reasoning with Chain-of-Thought (CoT)
   - **Modes Supported:**  
      - **pathology_reports:** Reads from `PathologyReports/`.
      - **consultation_notes:** Reads from `ConsultRedacted/`.
      - **treatment_plan_outcomepred:** Uses the original treatment plan prompt from combined reports in `PathConsCombined/`.
      - **path_consult_reports:** Uses a combined extraction prompt (merging pathology and consultation fields) from `PathConsCombined/`.
      - **CoT_treatment_plan_outcomepred:** Uses a chain‑of‑thought treatment plan prompt from `PathConsCombined/`.
   - **Key Feature:**  
   The script normalizes the report type input to lowercase and processes **only** the specified modes. It creates output subfolders for each mode processed.

- `combine_path_cons.py`: combine pathology reports & consultation notes texts inputs into one file per patient. 
- `run_local.sh`: testing scripts with ollama llama3.3:latest
- `run_local_all.sh`: Provides a convenient wrapper to run the Python script with a comma‑separated list of report types.  
   - input parent dir: /Data/Yujing/HNC_OutcomePred/HNC_Reports
   - results output dir: /Data/Yujing/HNC_OutcomePred/Reports_Agents_Results
      - **Usage Examples:**  
         - Process only CoT treatment plan mode:  
            ```bash
            ./run_local_all2.sh "CoT_treatment_plan_outcomepred"
            ```
         - Process both combined extraction and CoT mode:  
            ```bash
            ./run_local_all2.sh "path_consult_reports,CoT_treatment_plan_outcomepred"
            ```
         - Process all modes:  
            ```bash
            ./run_local_all2.sh all
            ```
      - **Key Feature:**  
      The script converts the input to lowercase, splits the report types into exact tokens, and sets booleans for each mode. The dynamic progress display shows only the counts for the requested modes. (Note: If using an output directory from a previous run, old subfolders may remain—consider using a clean output directory.)

### Organize Input Reports Data 
**Structure input directory as follows**
```
/media/yujing/One Touch3/HNC_Reports/
├── PathologyReports/       # .txt files for pathology reports
├── ConsultRedacted/        # .txt files for consultation notes
└── PathConsCombined/       # .txt files for combined reports (used in combined extraction and treatment plan modes)
```

### Configure JSON Prompts 
**Place all JSON prompt files in the propmts directory:**
```
/Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts/
  prompt_pathology_reports.json
  prompt_consultation_notes.json
  prompt_treatment_plan_outcomepred.json
  prompt_path_consult_reports.json
  prompt_CoT_treatment_plan_outcomepred.json
  ```

### Running the pipeline:
1) Optional: to run the `hnc_reports_agent2.py` script directly: 
```
python hnc_reports_agent.py \
  --prompts_dir /Data/Yujing/HNC_OutcomePred/Reports_Agents/prompts \
  --model_type local \
  --temperature 0.8 \
  --input_dir "/media/yujing/One Touch3/HNC_Reports" \
  --output_dir "/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp6" \
  --embedding_model ollama \
  --report_type CoT_treatment_plan_outcomepred \
  --local_model "llama3.3:latest"
```

2) Run the bash script, specify chosen report type(s): 
- `bash run_local_all2.sh "path_consult_reports, CoT_treatment_plan_outcomepred"`: runs 1) combined fields extractions from merged pathology reports & consultation notes, 2) CoT reasoning of treatment plan and outcome prediction 
- `bash run_local_all2.sh "path_consult_reports"`: only does 1) above 
... 


## Reuslts 
1. `/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp1 and Exp2`: initial experiment whose outputs reflected further prompt engineering suggestions by 3 students. 
2. `/Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/Exp3`: improved prompting for both pathology reports and consultation notes (less redundant, clear instruction on field inferences), and treatment plan and outcome prediction medical reasoning task (suboptimal performance; hallucinated due to prompting)
   - Reads separately the input pathology report, consultation notes for their respective extractions; the pathology report & consultation notes were combined into a single per patient txt file for the treatment plans & outcome prediction prompt. To run: 
      - `bash run_local_all.sh pathology_reports` only pathology reports
      - `bash run_local_all.sh consultation_notes` only consultation notes
      - `bash run_local_all.sh treatment_plan_outcomepred`only treatment plan
      - `bash run_local_all.sh all` to run all report types and prompts (3 above)
3. /Data/Yujing/HNC_OutcomePred/Reports_Agents_Results/[Exp_Number]
- ***Exp11***: pathology & consultation prompts and fields combined structured data outputs from combined path + cons text file per patient. This is due to sometimes the occurence of information fields existing in the other report types, decided to combine them. Also used to Chain-of-Thought prompting for the treatment plan & outcome predictions with references. 
- ***Exps5,6,7***: CoT medical reasoning examples 

## Evaluation Metrics 
- Human evaluation agreement metrics

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

### Consultation Notes

### Combined Pathology Reports & Consultation Notes for Medical Reasoning: 
- Using llama prompt engineering Chiain-of-Thought(CoT), which is simply to ask for adding a step-by-step thinking chain. 

### Privacy-Preserving Workflow
- Use local LLMs (e.g., Llama 3.3) on privacy-compliant infrastructure.[Meta Cookbook](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/Prompt_Engineering_with_Llama.ipynb); [Meta Prompt Eng](https://www.llama.com/docs/how-to-guides/prompting/); [npj Dig Med CoT and diagnostic reasoning prompts](https://www.nature.com/articles/s41746-024-01010-1/tables/2)

---
## Repository Structure
```
/Data/Yujing/HNC_OutcomePred/Reports_Agents/
├── input_reports/
│   ├── consultation_notes/
│   │   ├── consultation_notes_example1.txt
         ...
│   └── pathology_reports/
│       ├── pathology_report_example1.txt
│       ...
├── prompts/
│   ├── prompt_consultation_notes.json
│   └── prompt_pathology_report.json
├── output_dir/
│   ├── text_summaries/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_summary.txt
            ...
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_summary.txt
            ...
│   ├── embeddings/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_embedding.pkl
            ...
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_embedding.pkl
            ...
│   ├── structured_data/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_structured.csv
            ...
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_structured.csv
            ...
│   ├── structured_data_encoded/
│   │   ├── consultation_notes/
│   │   │   ├── patient_id1/
│   │   │   │   └── consultation_notes_structured_encoded.csv
            ...
│   │   └── pathology_reports/
│   │       ├── patient_id1/
│   │       │   └── pathology_reports_structured_encoded.csv
            ...
├── summaries_metadata.csv
├── tabular_data_metadata.csv
├── invalid_entries.log
└── processing_times.csv
```
---

## Future Work
1. **Extremely Early Fusion**
   - Combine clinical variable tabular data with reports before embedding generation.

2. **Integration with Multimodal Systems**
   - Extend the pipeline to include CT/PET images, dosimetry RT files, and histopathology slides.

3. **RAG and CoT Enhancements**
   - Evaluate Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) approaches for clinical predictions. RAG to reference from the NCCN Guidelines for Head and Neck Cancers. 

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

Feb 13th, 2025: 

- Python Script:
A single Python script (renamed to `hnc_reports_agent3.py`) that can process all cases or (with the --single flag) one random file per applicable subfolder. It also accepts a new parameter – --prompt_mode – so that you can choose between two prompt versions (for example, “combined” versus “default” or “separated”). In our example we include two JSON prompt files below.

- JSON Prompt Files:
Two versions are provided for the combined extraction task:

`prompt_path_consult_reports.json`: the default (simpler) version
`prompt_path_consult_reports_combined.json`: the version with detailed deduction instructions for fields that require reasoning (such as p16_Status/HPV_Status, Charlson_Comorbidity_Score, and performance scores).
Bash Scripts:
Two bash scripts are provided.

- `run_local_all3.sh`: runs the full‐processing mode (processing all cases)
- `run_prompt_experiment.sh`: runs in single‑case mode (processing one random file per applicable subfolder) for prompt engineering experiments. can optionally pass a case ID (via the environment variable CASE_ID or as part of the command line) so that the Python script will process that specific file. (If not given, it picks one random file per folder.)
Both scripts install a termination trap so that if you interrupt (Ctrl‑C), all child processes are killed.

Feb 14th, 2025 
# Two-Step Summarization Workflow (Agent4)

This README describes a **two-step** approach for summarizing and extracting structured data from **combined Pathology+Consultation** reports. The key idea is to use **two separate prompts**:
1. A **non-CoT** prompt for straightforward extraction.
2. A **chain-of-thought (CoT)** prompt for fields requiring inference (like **Charlson Comorbidity** scores, **Karnofsky**/ **ECOG** conversions, etc.).

Below are the main script and prompt files involved:

---

## 1. `hnc_reports_agent4.py`
**What it does**  
- Reads `.txt` files from:
  - `PathologyReports/` (pathology reports),
  - `ConsultRedacted/` (consultation notes),
  - `PathConsCombined/` (merged pathology + consultation).
- Summarizes them with a language model (local Llama, GPT, or Google PaLM) by:
  1. **One-step** approach for `pathology_reports`, `consultation_notes`, and so on, **except**:
  2. **Two-step** approach for `path_consult_reports`:  
     - Runs `prompt_path_consult_reports_extraction.json` first (non-CoT).  
     - Runs `prompt_path_consult_reports_cot.json` second (CoT).  
     - Merges both partial outputs into one final summary (one line per field).  
- Outputs:
  1. **Text Summaries** (`..._summary.txt`) in a standardized line-by-line format.  
  2. **Embeddings** (`..._embedding.pkl`) if an embedding model is configured.  
  3. (Optional) **Structured CSV** (`..._structured.csv`) and an **encoded** version for ML pipelines.  
  4. A global `processing_times.csv` with runtime stats.  

**Key Arguments**  
- `--report_type`: Which type(s) to process.  
- `--prompt_mode`: A suffix for your JSON prompts (e.g., `combined`).  
- `--case_id`: If provided, processes a single `.txt` that matches the case ID.  
- `--single`: If set, processes only one file per folder (random unless `case_id` is given).  
- `--model_type`: LLM backend (`local`, `gpt`, or `gemini`).  
- `--embedding_model`: Embedding model type (`ollama`, `openai`, or `google`).

**Example Usage (single case)**
```bash
python hnc_reports_agent4.py \
  --prompts_dir /Data/HNC/Agents/prompts \
  --model_type local \
  --temperature 0.8 \
  --input_dir "/media/hdd/HNC_Reports" \
  --output_dir "/Data/HNC/Results/Exp4" \
  --embedding_model ollama \
  --report_type "path_consult_reports" \
  --local_model "llama3.3:latest" \
  --prompt_mode "combined" \
  --single \
  --case_id "1130580"


---
