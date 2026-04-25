# 💳 GO Pinjam — Digital Credit Scoring Dashboard

> A Streamlit-based fintech dashboard that computes alternative credit scores from e-wallet transaction behaviour using an XGBoost risk model and Alibaba Qwen AI explanations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Data Format](#data-format)
- [Scoring Model](#scoring-model)
- [AI Explanation](#ai-explanation)

---

## Overview

GO Pinjam is a mobile-style credit scoring dashboard built for the Malaysian fintech context. It reads e-wallet transaction data (top-ups and payments), engineers six behavioural pillars, and maps them to a credit score (300–850) using a pre-trained XGBoost model. Users can view their score breakdown, transaction history, and generate a downloadable FinPassport report — with optional AI-powered explanations via Alibaba Qwen.

---

## Features

| Tab | What it does |
|---|---|
| 🏠 **Home** | Welcome screen with credit score ring, loan eligibility, risk badge, and recent transactions |
| 📊 **Score** | Full pillar breakdown, improvement tips, bar chart, and AI explanation |
| 📋 **History** | All transactions with spend/top-up summary, merchant diversity stats |
| 📄 **Report** | Animated FinPassport generation, downloadable `.txt` report, AI credit analysis |

---

## Project Structure

```
go-pinjam/
│
├── app.py                        # Main Streamlit application
├── alibaba_ai.py                 # Alibaba Qwen AI explanation module
├── transactions_cleaned.csv      # Raw transaction data (local fallback)
├── predictions_output.csv        # XGBoost model output (local fallback)
└── requirements.txt              # Python dependencies
```

---

## Requirements

**Python version:** 3.9+

**`requirements.txt`:**
```
streamlit
pandas
numpy
scipy
scikit-learn
boto3
botocore
openai
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Installation

```bash
# 1. Clone or download the project
git clone <your-repo-url>
cd go-pinjam

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data files (see Data Format below)
# Place transactions_cleaned.csv and predictions_output.csv in the project root

# 4. Configure your Alibaba API key in alibaba_ai.py (see AI Explanation below)
```

---

## Configuration

### Local Mode (default)
No configuration needed. Just place both CSV files in the same folder as `app.py` and run.

### AWS S3 Mode (optional)
To load data from S3, edit these two lines in `app.py`:

```python
AWS_REGION = "ap-southeast-1"     # Your AWS region
S3_BUCKET  = "your-bucket-name"   # Your S3 bucket (anything other than "tngdgo-pinjam")
```

Then upload your files to:
```
s3://your-bucket-name/input/transactions_cleaned.csv
s3://your-bucket-name/output/predictions_output.csv
```

AWS credentials must be configured via `~/.aws/credentials`, environment variables, or an IAM role.

---

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

**Login (Demo):**
- Select any User ID from the dropdown
- Enter any email and password (no real auth — demo mode)

---

## Data Format

### `transactions_cleaned.csv`

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Unique user identifier (e.g. `USER_00001`) |
| `transaction_date` | datetime | Date of transaction |
| `transaction_type` | string | `payment` or `topup` |
| `product_amount` | float | Transaction amount in RM |
| `product_category` | string | Merchant category (e.g. `Food Delivery`) |
| `merchant_name` | string | Merchant name |
| `transaction_status` | string | `Successful` or `Failed` |

### `predictions_output.csv`

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Unique user identifier |
| `predicted_risk` | int | `0` = Low Risk, `1` = High Risk |
| `risk_label` | string | `Low Risk` or `High Risk` |
| `credit_score` | int | Score between 300–850 |
| `risk_tier` | string | `Poor`, `Fair`, `Good`, or `Very Good` |
| `score_spending` | float | Pillar sub-score 0–1 |
| `score_frequency` | float | Pillar sub-score 0–1 |
| `score_diversity` | float | Pillar sub-score 0–1 |
| `score_reliability` | float | Pillar sub-score 0–1 |
| `score_maturity` | float | Pillar sub-score 0–1 |
| `score_topup` | float | Pillar sub-score 0–1 |

---

## Scoring Model

Credit scores are derived from **six behavioural pillars** engineered from raw transaction data:

| Pillar | Weight | Key Signals |
|---|---|---|
| 🔄 Top-up Consistency | 25% | Frequency, stability, top-up-to-spend ratio |
| 💸 Spending Behaviour | 20% | Monthly spend, payment stability |
| 📅 Transaction Frequency | 20% | Transactions/month, active months ratio, recency |
| 🏪 Merchant Diversity | 15% | Unique categories, category entropy |
| ✅ Reliability | 10% | Failed transaction rate, success rate |
| 📆 Account Maturity | 10% | Total months of history |

All features are normalised **globally across all users** using MinMaxScaler before scoring.

**Score → Risk Tier mapping:**

| Score Range | Tier | Loan Limit |
|---|---|---|
| 660–850 | Very Good | RM 15,000 |
| 590–659 | Good | RM 10,000 |
| 520–589 | Fair | RM 5,000 |
| 300–519 | Poor | RM 1,000 |

---

## AI Explanation

Powered by **Alibaba Qwen** (`qwen-plus`) via OpenAI-compatible API.

The `alibaba_ai.py` module sends a structured payload to Qwen and returns a short markdown-style credit explanation covering:
- Why the score was assigned
- Key contributing factors
- Three actionable improvement tips
- An encouraging closing statement

**To configure**, update `alibaba_ai.py` with your API key and endpoint:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-alibaba-api-key",
    base_url="https://your-endpoint.aliyuncs.com/compatible-mode/v1"
)
```

If the AI service is unavailable, the app gracefully falls back to a warning message without crashing.

---

## Notes

- The login screen is **demo-only** — any email/password combination works
- All feature engineering and normalisation happens at startup and is cached by Streamlit (`@st.cache_data`)
- The `USE_S3` flag auto-detects S3 mode based on whether `S3_BUCKET` differs from the default placeholder value
