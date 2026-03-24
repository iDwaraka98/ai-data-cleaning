# Leveraging Generative AI for Automated Data Cleaning & Error Correction

**Author:** Dwarakamai Illipilla  
**Degree:** MSc Data Science  
**Dataset:** US Retail Sales Dataset (9,641 records)

---

## Problem Statement

Data quality is a fundamental requirement for reliable data analysis and decision-making. However, real-world datasets often contain inconsistencies such as missing values, duplicates, formatting errors, and outliers, which can negatively impact analytical outcomes.

Traditional data cleaning approaches are primarily rule-based and require significant manual effort and domain knowledge. These methods are often limited in handling complex or context-dependent errors and may not scale efficiently with increasing data volume.

This study addresses the challenge of automating the data cleaning process by proposing an AI-based system capable of detecting and correcting multiple types of data quality issues. The system aims to enhance data reliability while reducing manual intervention.

Furthermore, the effectiveness of the proposed approach is evaluated through comparison with a conventional rule-based method to assess its performance and practical applicability.

---

## Project Structure

```
ai_data_cleaning/
│
├── data/
│   └── data_sales.csv               ← Place your dataset here
│
├── modules/
│   ├── error_detector.py            ← Module 1: Error Detection
│   ├── llm_corrector.py             ← Module 2: LLM Correction Generation
│   ├── validator.py                 ← Module 3: Validation & Confidence Scoring
│   └── utils.py                     ← Helper functions
│
├── notebooks/
│   └── EDA.ipynb                    ← Exploratory Data Analysis notebook
│
├── outputs/
│   └── (cleaned files saved here)
│
├── tests/
│   └── test_pipeline.py             ← Unit tests
│
├── pipeline.py                      ←  Main entry point — run this
├── baseline.py                      ← Rule-based baseline for comparison
├── evaluate.py                      ← Evaluation & comparison metrics
├── requirements.txt
└── README.md
```

---


## Features

Automated detection of multiple data quality issues, including missing values, duplicates, and inconsistencies
AI-based correction mechanism for intelligent data cleaning
Multi-layer validation to ensure reliability of corrections
Comparative analysis with traditional rule-based methods
Generation of evaluation metrics and visual insights
Modular system design for scalability and maintainability

---

##  Dataset Description

Transaction ID – Unique identifier for each transaction
Date – Transaction date
Customer ID – Identifier for customers
Product Category – Type of product purchased
Quantity – Number of items sold
Price – Unit price of the product
Total Sales – Total transaction value
Payment Method – Mode of payment
Store Location – Location of the transaction

---

---

