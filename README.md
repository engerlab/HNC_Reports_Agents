# HNC clinical reports agents for extraction key info

# Created a Github repository for the project named HNC_Reports_Agents:

cd /Data/Yujing/HNC_OutcomePred/Reports_Agents
git init
git remote add origin https://github.com/yujing1997/HNC_Reports_Agents.git
git add .
git commit -m "Initial commit"
git push -u origin master

## Create virtual environment for this 
`python3 -m venv env`
`source env/bin/activate`
or
`conda activate segment2`
`conda install langchain chromadb openai google-generative-ai tqdm pandas`
Add other necessary installations based on your models and embeddings

## Package requirements quirks:
`conda install pip`
`pip install --upgrade pip`, `pip install -U langchain-ollama`, `pip show langchain-ollama`
`pip install -qU langchain-openai`



## Relavant literature: 1) unstructured text to embedding; 2) unstructured to structured text to embedding
    - Expert-guided knowledge for something oncology (JCO Precision Oncology)
    - LLM-For-Survival-Analysis (https://github.com/yujing1997/LLM-For-Survival-Analysis/tree/main)
    - See Google slides notes

## Purpose: transforming unstructured reports to more structured data
    - key summary (texts)
    - structured variables 
    - expert-guided knowledge for prompt engineering needed

## Llama LLM text summary & tabular data extraction
    - Originally inspired by the AG part of RAG: https://github.com/engerlab/medphys-agent/blob/main/langchain/stream/qa.py
    - Yujing/Laya/Farhood: adapt, reports are on Proton 
    - Prompt engineering: me, Drs. Khalil Sultanem & George Shenouda
    - Turn into text summary → embedding; text summary → expert-guided variables (quality-control needed)
    - Run text summary embeddings (post-Llama) → ClinicalLongformer on previous pipeline

## Expert knowledge requested for prompt engineering: 
- Categorical answer: yes/no/cannot infer 
- Categorical answer: need to already know what the answers are; grade 1, grade 2, grade 3… expert comes 
- Numerical answer: ask specific numbers  
- Are there variations you'd expect from these reports? 


## Development of Clinical reports agents workflow 

1. Document Ingestion
- Collect .txt files containing pathology reports and consultation notes from a specified input directory.
2. Prompt Configuration/Engineering 
- Utilize a separate prompts.json file to define distinct prompts for each report type (e.g., pathology reports, consultation notes) to facilitate flexible and targeted summarization.
3. Model Initialization
- Initialize the appropriate language model (Llama 3.2 or Llama 3.3. via Ollama, OpenAI GPT, or Google Gemini) based on user selection, configuring parameters such as temperature for response variability.
4. Summary Generation
- For each report, generate a concise summary containing only the key information by feeding the report text and the corresponding prompt into the language model using LangChain’s framework.
5. Quality Control
- Implement automated validation checks to ensure that the generated summaries include all required fields and adhere to the expected format. Log any discrepancies or failures for manual review.
6. Post-Processing
- Extract structured data from the summaries using regular expressions tailored to each report type, converting unstructured text into organized tabular variables.
7. Normalization
- Standardize the extracted data by converting units (e.g., tumor sizes to millimeters), handling missing values, and ensuring consistency across all data entries.
8. Embedding Generation
- Generate embeddings for each summary using specified embedding models (e.g., OpenAI Embeddings, Ollama Embeddings, Google Generative AI Embeddings) to facilitate downstream tasks such as clustering or similarity searches.
9. Output Storage
- Save the generated summaries, extracted tabular data, normalized data, and embeddings into designated output directories in appropriate formats (e.g., CSV, pickle files).
10. Logging and Error Handling
- Maintain comprehensive logs detailing the processing status of each report, including successful summaries and any encountered errors or validation failures.

### Input Reports
- pathology report example taken from
    - /media/yujing/One Touch3/HNC_Reports/PathologyReports/1244601.txt
- consultation note example taken from
    - /media/yujing/One Touch3/HNC_Reports/ConsultRedacted/1259610.txt

### Prompts 
- Each JSON file contains a "prompts" key with a list of different prompts tailored to the respective report type.
- {context} is a placeholder that will be replaced with the actual report text during processing.


Note:
- ollama run llama3.3 ongoing... if not finished downaloading in the linux terminal, just run this command again, it will continue downloading from where it stopped. 

### Project directory 
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
└── output_dir/
    └── ...

### Input Directory Structure 

### Desired Output Directory Structure
results/
├── text_summaries/
│   ├── 129701/
│   │   ├── consultation_notes_summary.txt
│   │   └── pathology_report_summary.txt
│   └── ...
├── embeddings/
│   ├── 129701/
│   │   ├── consultation_notes_embedding.pkl
│   │   └── pathology_report_embedding.pkl
│   └── ...
├── structured_data/
│   ├── 129701/
│   │   ├── consultation_notes_structured.csv
│   │   └── pathology_report_structured.csv
│   └── ...
└── structured_data_encoded/
    ├── 129701/
    │   ├── consultation_notes_structured_encoded.csv
    │   └── pathology_report_structured_encoded.csv
    └── ...


#### Notes on pathology reports:
- some were done as "Peroperatoire-Intraoperative Consultation"
    - which means that it was taken during the surgery, sent to pathology labs, from which the real accurate lymph status (positive or negative) and the number of them can be determined. Exmpale: /Data/Yujing/HNC_OutcomePred/Reports_Agents/output_dir/text_summaries/44651/pathology_reports_summary.txt

#### Prompt Engineering
##### Pathology reports modifications:
- if I want structured data to encode the following for the summarized pathology report: quantitative or numerical metrics, atient history and status, anatomic site of lesion, 
... lymph node status: presence/absence, number of lymph nodes, extranodal extension, resection margin: positive or negative, HPV or p16 status: positive or negative, smoking hito
... ry: never smoked, or 10 years smoking 1 pack a day or more than 10 years, or alcohol consultion. do not make up any information. structure the response. if something was not  
... inferred from the original report, say not inferred. here's the summarized pathology report: This is a final pathology report for an 88-year-old woman with left alveolar cance
... r.
