
# FetalCare AI

### Early Detection of Fetal Distress Using Machine Learning

---

## Overview

**FetalCare AI** is a clinical decision-support system designed to assist doctors in detecting fetal hypoxia early, using data-driven insights from **cardiotocography (CTG) signals**.
Built with empathy and precision, the system integrates **CatBoost-based predictions** with real-time inference through **FastAPI** and a **Streamlit clinical dashboard**, deployed on **AWS Bedrock** for scalability and reliability.

---

## Inspiration

During a visit to a special child centre in Bangalore, I met children who could neither walk nor crawl.
When I asked the physiotherapist what had caused it, she said quietly:

> “Most of them didn’t get enough oxygen during birth.”

That moment changed me. These children weren’t born with limitations — they were failed by delayed detection.
No mother should lose her child to preventable hypoxia.

I built **FetalCare AI** to help doctors identify fetal distress before it’s too late — giving every unborn child a fair start in life.

---

## Abstract

This project applies **machine learning** to classify fetal states as *Normal*, *Suspect*, or *Pathologic* using CTG features such as fetal heart rate, uterine contractions, accelerations, and decelerations.
The model emphasizes **explainability**, **real-time inference**, and **scalability**, ensuring that clinicians can interpret and act on predictions with confidence.

**Key Outcomes:**

* Improved interpretability through **SHAP visualizations**
* Robust performance from **CatBoost** with balanced data
* AWS-integrated, low-latency clinical deployment

---

## Technical Stack

| Component          | Technology                                   |
| ------------------ | -------------------------------------------- |
| **Dataset**        | UCI Cardiotocography (2,126 labeled samples) |
| **Model**          | CatBoost Classifier                          |
| **Backend**        | FastAPI (Python)                             |
| **Frontend**       | Streamlit / React                            |
| **Deployment**     | AWS Bedrock and serverless stack             |
| **Explainability** | SHAP feature importance visualization        |

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Yaaseen-Basit/fetalcare_ai.git
cd fetalcare_ai
```

Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

Start backend (FastAPI):

```bash
uvicorn backend.agent_backend:app --reload
```

Start frontend (Streamlit UI):

```bash
streamlit run frontend/ui.py
```

---

## Features

* Predicts fetal risk level (Normal, Suspect, Pathologic)
* Real-time inference via FastAPI endpoint
* SHAP explainability to support clinical interpretation
* Intuitive, responsive UI for healthcare professionals
* AWS Bedrock deployment for scalability

---

## Challenges

* **Data Imbalance:** Balanced normal and pathological cases through resampling
* **Explainability:** Integrated SHAP for clinician trust
* **Time Constraints:** Built and deployed under hackathon conditions
* **Emotional Balance:** Combined empathy with technical precision

---

## Impact

FetalCare AI is designed to support early detection of fetal hypoxia — particularly in **low-resource clinics and rural areas** where advanced monitors are scarce.
Even if it helps **one doctor make a timely decision** or **one mother save her child**, it has fulfilled its purpose.

---

## Vision

To make AI-powered fetal monitoring **accessible**, **interpretable**, and **affordable** for every healthcare center — ensuring that no child’s first battle for oxygen goes unheard.

**Every heartbeat matters. Every life deserves a fair beginning.**

---
## Reference

If you refer to this project, please cite it as:

Yaaseen Basit, *FetalCare AI – Early Detection of Fetal Distress Using Machine Learning*, GitHub Project, 2025.  
Developed during the AWS AI Agent Global Hackathon 2025.

**Note:** This project uses the UCI Cardiotocography (CTG) dataset for research and academic purposes.  
Dataset: UCI Cardiotocography (CTG), used for academic and research purposes.