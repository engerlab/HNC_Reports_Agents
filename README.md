# HNC clinical reports agents for extraction key info

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


