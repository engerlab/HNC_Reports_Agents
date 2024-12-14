# HNC clinical reports agents for extraction key info

# Created a Github repository for the project named HNC_Reports_Agents:

cd /Data/Yujing/HNC_OutcomePred/Reports_Agents
git init
git remote add origin https://github.com/yujing1997/HNC_Reports_Agents.git
git add .
git commit -m "Initial commit"
git push -u origin master

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
- Initialize the appropriate language model (Llama 3.2 via Ollama, OpenAI GPT, or Google Gemini) based on user selection, configuring parameters such as temperature for response variability.
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

### Methodology (for manuscript)
We developed an automated pipeline to summarize pathology reports and consultation notes, extracting key clinical information for further analysis. The workflow commenced with the ingestion of .txt files containing unstructured pathology reports and consultation notes from a designated input directory. To facilitate targeted summarization, a separate configuration file (prompts.json) was employed to define specific prompts tailored to each report type, enabling flexibility and ease of modification.

The initialization of the language model was contingent upon user selection, supporting models such as Llama 3.2 via Ollama, OpenAI’s GPT-4, and Google’s Gemini 1.5-flash. Parameters like temperature were configurable to control the variability and creativity of the generated summaries. Utilizing LangChain’s framework, each report was processed by feeding its text alongside the corresponding prompt into the selected language model, resulting in concise summaries that encapsulated only the essential information.

To ensure the reliability and accuracy of the summaries, an automated quality control mechanism was implemented. This involved validating the presence and correct formatting of all required fields within each summary using predefined regular expressions. Any summaries failing these checks were logged for manual inspection, ensuring that only high-quality data proceeded to subsequent stages.

The post-processing phase involved extracting structured data from the validated summaries. This was achieved through the application of regular expressions designed to capture specific fields such as Diagnosis, Tumor Size, Grade, and Biomarkers for pathology reports, and Patient Concerns, Recommendations, and Follow-up Actions for consultation notes. The extracted data was then normalized to maintain consistency, which included converting tumor sizes to a uniform unit of millimeters, handling missing values, and standardizing categorical data entries.

Subsequently, embeddings were generated for each summary to facilitate advanced analytical tasks such as clustering and similarity searches. Depending on the user's choice, embedding models like OpenAI Embeddings, Ollama Embeddings, or Google Generative AI Embeddings were utilized. These embeddings were stored in a pickle file (summary_embeddings.pkl) for efficient retrieval and future use.

Finally, all processed data, including the generated summaries, extracted and normalized tabular data, and embeddings, were saved into designated output directories in appropriate formats (e.g., CSV for tabular data and summaries, pickle for embeddings). Comprehensive logging ensured that the entire process was transparent and that any errors or validation failures were systematically recorded for subsequent review.

This methodology provided a robust and scalable solution for transforming unstructured clinical reports into structured, analyzable data, thereby facilitating enhanced data management and supporting downstream clinical and research applications.

