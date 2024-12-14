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
`pip install langchain chromadb openai google-generative-ai tqdm pandas`
Add other necessary installations based on your models and embeddings


## Relavant literature:
    - Expert-guided knowledge for something oncology (JCO Precision Oncology)
    - See Google slides notes

## Purpose: transforming unstructured reports to more structured data
    - key summary (texts)
    - structured variables 
    - expert-guided knowledge for prompt engineering needed

## Llama LLM text summary & tabular data extraction
    - https://github.com/engerlab/medphys-agent/blob/main/langchain/stream/qa.py
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

### input_reports
- pathology report example taken from
    - /media/yujing/One Touch3/HNC_Reports/PathologyReports/1244601.txt
- consultation note example taken from
    - /media/yujing/One Touch3/HNC_Reports/ConsultRedacted/1259610.txt

### prompts 
- Each JSON file contains a "prompts" key with a list of different prompts tailored to the respective report type.
- {context} is a placeholder that will be replaced with the actual report text during processing.