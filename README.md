# FairXAI: A Framework for Bias Detection and Mitigation in Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project introduces **FairXAI**, a pipeline that integrates fairness-aware techniques with explainable AI (XAI) to detect and mitigate bias in machine learning models. The pipeline combines:

- **Fairness Metrics** (Equal Opportunity, Demographic Parity, Disparate Impact, Error Rate Parity)  
- **Bias Mitigation** (Reweighting technique by Kamiran & Calders)  
- **Explainability** (SHAP and LIME for global/local interpretation)  
- **Natural Language Explanations** (using Phi-2, a small LLM for human-readable insights)

This work is applied to real-world datasets in criminal justice and income prediction and is designed for easy adaptation to other domains.



## Datasets

### 1. [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Predict whether a person earns >$50K annually
- Bias focus: Gender and Race

### 2. [COMPAS Recidivism Dataset](https://github.com/propublica/compas-analysis)
- Predict likelihood of criminal recidivism
- Bias focus: Racial disparities


---

## Features

- **Fairness Evaluation**  
  Quantitative analysis using fairness metrics on pre- and post-mitigation models

- **Bias Mitigation**  
  Reweights training data to address underrepresentation and reduce prediction skew

- **Explainability**  
  - SHAP for global/local feature importance
  - LIME for individual prediction inspection

- **Natural Language Reasoning with Phi-2**  
  Translates model behavior into simple, understandable summaries for non-technical users

---

## Results

**Before Mitigation (Adult Dataset)**  
- True Positive Rate (TPR): 0.45 (Women), 0.81 (Men)  
- Disparate Impact: 0.37  

**After Mitigation (Adult Dataset)**  
- TPR: 0.83 (Women), 0.87 (Men)  
- Disparate Impact: 0.78  

**Phi-2 Output Example**  
> "This individual is predicted to earn >$50K primarily due to their strong educational background and managerial role, with gender having a moderate influence."

---

## Installation

### 1. Clone the Repository

git clone https://github.com/akinahomwabella/fair_xai_pipeline.git
cd fair_xai_pipeline

### 2.Create Virtual Environment
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows

### 3. Install Dependencies
pip install -r requirements.txt
@misc{akinahom_fairxai_2025,
  title={A Framework for Bias Detection and Mitigation in Machine Learning},
  author={Wabella, Akinahom},
  year={2025},
  url={https://github.com/akinahomwabella/fair_xai_pipeline}
}
