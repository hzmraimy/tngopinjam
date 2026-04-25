# TNGo Pinjam

A Streamlit app for loan prediction and user dashboard based on transaction data.

## Features

- User credit score calculation
- Risk assessment
- Transaction history visualization
- Loan eligibility determination

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Deployment

This app is deployed on Streamlit Cloud.

### Files to include in repo:

- `app.py` - Main application
- `requirements.txt` - Dependencies
- `transactions_cleaned.csv` - Transaction data
- `predictions_output.csv` - Precomputed predictions
- `finalized/` - Model and notebook (optional)

### Files to exclude:

- Virtual environments (`myenv/`, `myenv311/`)
- Cache files (`__pycache__/`, `.DS_Store`)

## Data

The app uses preprocessed transaction data and precomputed model predictions. Ensure data files are in the same directory as `app.py`.