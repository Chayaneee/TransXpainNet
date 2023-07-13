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

# Evaluation Metrics 
1. Word Overlap Metrics: BLEU-score, METEOR, ROUGE-L, CIDER
2. Clinical Efficiency (CE) Metrics: AUC, F1-score, Precision, Recall, Accuracy
3. Semantic Similarity-based Metrics: Skip-Thoughts, Average Embedding, Vector Extrema, Greedy Matching

# Quantative Results

| $textbf{Datasets}$                     | $\textbf{Models}$                                           | $\textbf{B1}$         | $\textbf{B2}$          | $\textbf{B3}$         | $\textbf{B4}$         | $\textbf{METEOR}$     | $\textbf{ROUGE-L}$    | $\textbf{CIDER}$      |
|-------------------------------------|-----------------------------------------------------------|---------------------|----------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| IU X-ray  | Sonit et al. 2019                     | $0.374$             | $0.224$              | $0.152$             | $0.11$              | $0.164$             | $0.308$             | $0.360$             |
|                                     | Xu et al. 2018                    | $0.464$             | $0.358$              | $0.270$             | $0.195$             | $\textbf{0.274}$    | $0.366$             | $--$                |
|                                     | Jing et al. 2018           | $\textbf{0.517}$    | $\textbf{0.386}$     | $\textbf{0.306}$    | $\textbf{0.247}$    | $0.217$             | $\textbf{0.447}$    | $\underline{0.327}$ |
|                                     | Omar et al. 2021            | $0.387$             | $0.245$              | $0.166$             | $0.111$             | $0.164$             | $0.289$             | $0.257$             |
|                                    | R2GEN 2020                      | $0.470$             | $0.304$              | $0.219$             | $0.165$             | $0.187$             | $0.371$             | $--$                |
|                                   | Xiong et al. 2019              | $0.350$             | $0.234$              | $0.143$             | $0.096$             | $--$                | $--$                | $0.323$             |
|                                     | Nguyen et al. 2021             | $\underline{0.515}$ | $\underline{0.378}$ | $\underline{0.293}$ | $\underline{0.235}$ | $\underline{0.219}$ | $\underline{0.436}$ | $--$                |
|                                     | R2GENCMN 2022                        | $0.475$             | $0.309$              | $0.222$             | $0.170$             | $0.191$             | $0.375$             | $--$                |
|                                     | Ours                                                      | $0.483$             | $0.352$              | $0.273$             | $0.219$             | $0.208$             | $0.418$             | $\textbf{0.536}$    |
| MIMIC-CXR | Liu et. al                       | 0.313               | 0.206                | 0.146               | 0.103               | --                  | 0.306               | --                  |
|                                   | 1-NN 2020                       | 0.367               | 0.215                | 0.138               | 0.095               | 0.139               | 0.228               | --                  |
|                                 | R2GEN 2020                      | 0.353               | 0.218                | 0.145               | 0.103               | 0.142               | 0.277               | --                  |
|                                     | Transformer Prog. 2021 | 0.378               | 0.232                | 0.154               | 0.107               | 0.145               | 0.272               | --                  |
|                                   | PPKED 2021                        | 0.360               | 0.224                | 0.149               | 0.106               | 0.149               | 0.284               | --                  |
|                                     | Co-ATT 2021             | 0.350               | 0.219                | 0.152               | 0.109               | 0.151               | 0.283               | --                  |
|                                     | Nguyen et al. 2021             | $\textbf{0.495}$      | $\textbf{0.360}$       | $\textbf{0.278}$      | $\textbf{0.224}$      | $\textbf{0.222}$      | $\textbf{0.390}$      | --                  |
|                                     | R2GENCMN 2022                         | 0.353               | 0.218                | 0.148               | 0.106               | 0.142               | 0.278               | --                  |
|                                     | XPRONET 2022                         | 0.344               | 0.215                | 0.146               | 0.105               | 0.138               | 0.279               | --                  |
|                                     | CvT-212DistilGPT2 2022        | 0.395               | 0.249                | 0.172               | 0.127               | 0.155               | 0.288               | $\textbf{0.379}$      |
|                                     | Ours (with $25\%$)                                        | $\underline{0.432}$   | $\underline{0.296}$   | $\underline{0.218}$   | $\underline{0.167}$   | $\underline{0.181}$   | $\underline{0.336}$   | $\underline{0.272}$   |




# Qualitative Results
# 1. DeiT-CXR 
![DeiT-CXR_Results](https://github.com/Chayaneee/Transformers_Based_Automatic_Report_Generation/assets/54748679/4fcb6510-92b9-4534-8328-3943d55de7c4)

# 2. Report Generation
![Report_Generation](https://github.com/Chayaneee/Transformers_Based_Automatic_Report_Generation/assets/54748679/c2a6ccf4-c0f4-4d55-85d0-1034269eeb47)

