# Transformers_Based_Automatic_Report_Generation
Transformers Are All You Need to Generate Automatic Report from Chest X-ray Images


# Introduction: Transformers_Based_Automatic_Report_Generation
This is the official repository of our proposed Fully Transformers_Based_Automatic_Report_Generation model details. Chest X-ray imaging is crucial for diagnosing and treating thoracic diseases, but the process of examining and generating reports for these images can be challenging. There is a shortage of experienced radiologists, and report generation is time-consuming, reducing effectiveness in clinical settings. To address this issue and advance clinical automation, researchers have been working on automated systems for radiology report generation. However, existing systems have limitations, such as disregarding clinical workflow, ignoring clinical context, and lacking explainability. This paper introduces a novel model for automatic chest X-ray report generation based entirely on transformers. The model focuses on clinical accuracy while improving other text-generation metrics. It utilizes a domain-knowledge-based vision transformer called DeiT-CXR to extract image features and incorporates supportive documents like clinical history to enhance the report generation process. The model is trained and tested on two X-ray report generation datasets, IU X-ray and MIMIC-CXR, demonstrating promising results regarding word overlap, clinical accuracy, and semantic similarity-based metrics. Additionally, qualitative results using Grad-CAM showcase disease location for better understanding by radiologists. The proposed model embraces radiologists' workflow, aiming to improve explainability, transparency, and trustworthiness for their benefit.

# Proposed Pipeline
![Block_Diagram](https://github.com/Chayaneee/Transformers_Based_Automatic_Report_Generation/assets/54748679/145254f7-1e4f-4b24-85e8-c0edf9e60a1b)


# Data used for experiments: 

We have used three datasets for this experiment.
  - [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
  - [IU X-ray](https://openi.nlm.nih.gov/)
  - [MIMIC-CXR](https://physionet.org/content/mimiciii-demo/1.4/)
  
# Qualitative Results
# 1. DeiT-CXR 
