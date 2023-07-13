# Transformers_Based_Automatic_Report_Generation
Transformers Are All You Need to Generate Automatic Report from Chest X-ray Images


# Introduction: Transformers_Based_Automatic_Report_Generation
This is the official repository of our proposed Fully Transformers_Based_Automatic_Report_Generation model details. Chest X-ray imaging is crucial for diagnosing and treating thoracic diseases, but the process of examining and generating reports for these images can be challenging. There is a shortage of experienced radiologists, and report generation is time-consuming, reducing effectiveness in clinical settings. To address this issue and advance clinical automation, researchers have been working on automated systems for radiology report generation. However, existing systems have limitations, such as disregarding clinical workflow, ignoring clinical context, and lacking explainability. This paper introduces a novel model for automatic chest X-ray report generation based entirely on transformers. The model focuses on clinical accuracy while improving other text-generation metrics. It utilizes a domain-knowledge-based vision transformer called DeiT-CXR to extract image features and incorporates supportive documents like clinical history to enhance the report generation process. The model is trained and tested on two X-ray report generation datasets, IU X-ray and MIMIC-CXR, demonstrating promising results regarding word overlap, clinical accuracy, and semantic similarity-based metrics. Additionally, qualitative results using Grad-CAM showcase disease location for better understanding by radiologists. The proposed model embraces radiologists' workflow, aiming to improve explainability, transparency, and trustworthiness for their benefit.



<p align="center">
  <img src="https://github.com/ginobilinie/xray_report_generation/blob/main/img/motivation.png" width="400" height="400">
</p>


# Data used for experiments: 

We have used three datasets for this experiment.
  - [CheXpert]([https://openi.nlm.nih.gov/](https://stanfordmlgroup.github.io/competitions/chexpert/))
  - [IU X-ray](https://openi.nlm.nih.gov/)
  - [MIMIC-CXR](https://physionet.org/content/mimiciii-demo/1.4/)
  
