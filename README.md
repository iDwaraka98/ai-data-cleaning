# 🤖 Leveraging Generative AI for Automated Data Cleaning & Error Correction

**Author:** Dwarakamai Illipilla  
**Degree:** MSc Data Science  
**Dataset:** US Retail Sales Dataset (9,641 records)

---

## 📌 Problem Statement

Real-world retail datasets suffer from missing values, typographical errors, format
inconsistencies, logical mismatches, and outliers. Traditional rule-based methods
cannot handle errors that require contextual understanding. This project builds a
complete AI-driven pipeline using GPT-4 to automatically detect and correct these
errors, validated by a multi-layer confidence scoring system.

---

## 🗂️ Project Structure

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
├── pipeline.py                      ← ⭐ Main entry point — run this
├── baseline.py                      ← Rule-based baseline for comparison
├── evaluate.py                      ← Evaluation & comparison metrics
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai_data_cleaning.git
cd ai_data_cleaning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your OpenAI API Key
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```
> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 4. Add Your Dataset
Place your CSV file in the `data/` folder and name it `data_sales.csv`.

### 5. Run the Pipeline
```bash
python pipeline.py
```

### 6. Run Without OpenAI (Rule-Based Baseline Only)
```bash
python baseline.py
```

### 7. Run Evaluation & Comparison
```bash
python evaluate.py
```

---

## 🔑 Features

| Feature | Description |
|---|---|
| ✅ Error Detection | Missing values, typos, format issues, duplicates, logical errors, outliers |
| 🤖 LLM Correction | GPT-4 with zero-shot, few-shot, chain-of-thought, and RAG strategies |
| 🛡️ Validation | 3-layer validation: format, logical consistency, statistical plausibility |
| 📊 Confidence Scoring | Auto-apply ≥0.80, flag 0.50–0.79, human review <0.50 |
| 📈 Evaluation | Precision, Recall, F1 vs rule-based baseline |
| 🔍 Explainability | SHAP analysis of correction reliability |

---

## 📊 Dataset Columns

| Column | Type | Notes |
|---|---|---|
| Retailer | Categorical | 6 unique retailers |
| Retailer ID | Integer | |
| Invoice Date | Date | MM/DD/YYYY format |
| Region | Categorical | 5 US regions |
| State | Categorical | 50 states |
| City | Categorical | 52 cities |
| Product | Categorical | 7 product types (contains typo) |
| Price per Unit | String/Float | Contains $ symbol and spaces |
| Units Sold | String/Int | May contain commas |
| Total Sales | String/Float | Contains $, commas |
| Operating Profit | String/Float | Contains $, commas |
| Sales Method | Categorical | Online / Outlet / In-store |

---

## 🧪 Running on Google Colab

```python
# Cell 1: Clone and setup
!git clone https://github.com/YOUR_USERNAME/ai_data_cleaning.git
%cd ai_data_cleaning
!pip install -r requirements.txt

# Cell 2: Set API key
import os
os.environ["OPENAI_API_KEY"] = "your_key_here"

# Cell 3: Run pipeline
!python pipeline.py
```

---

## 📄 License
MIT License — free to use for academic and research purposes.
