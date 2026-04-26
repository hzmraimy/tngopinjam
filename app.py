import streamlit as st
import time
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from alibaba_ai import get_explanation
import boto3
from botocore.exceptions import NoCredentialsError

# ─── S3 Configuration ─────────────────────────────────────────────────────────
# Change these to your AWS region and bucket name
AWS_REGION = "ap-southeast-1"  # Change to your region
S3_BUCKET = "tngdgo-pinjam"  # Change to your S3 bucket


# S3 file paths
TRANSACTIONS_CSV = f"s3://{S3_BUCKET}/input/transactions_cleaned.csv"
PREDICTIONS_CSV  = f"s3://{S3_BUCKET}/output/predictions_output.csv"

# Fallback to local files if S3 not configured
BASE_DIR = Path(__file__).parent
#LOCAL_TRANSACTIONS = BASE_DIR / 'transactions_cleaned.csv'
#LOCAL_PREDICTIONS  = BASE_DIR / 'predictions_output.csv'

USE_S3 = S3_BUCKET != "tngdgo-pinjam"  # Auto-detect if S3 is configured

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"GO Pinjam{USE_S3}",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── S3 Helper Functions ──────────────────────────────────────────────────────
def get_s3_client():
    """Create S3 client with credentials from environment or AWS config"""
    try:
        return boto3.client('s3', region_name=AWS_REGION)
    except NoCredentialsError:
        st.error("❌ AWS credentials not found. Configure AWS credentials.")
        st.stop()

@st.cache_data
def load_csv_from_s3(s3_path: str):
    """Load CSV directly from S3 using pandas"""
    try:
        return pd.read_csv(s3_path)
    except Exception as e:
        st.error(f"❌ Failed to load {s3_path}: {str(e)}")
        st.stop()

@st.cache_data
def load_csv_from_local(file_path):
    """Load CSV from local file system"""
    if not file_path.exists():
        st.error(f"❌ File not found: {file_path}\n\nMake sure `{file_path.name}` is in the same folder as `app.py`.")
        st.stop()
    return pd.read_csv(file_path)

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if USE_S3:
        # Load from S3
        df = load_csv_from_s3(TRANSACTIONS_CSV)
        pred = load_csv_from_s3(PREDICTIONS_CSV)
    # else:
    #     # Load from local files
    #     df = load_csv_from_local(LOCAL_TRANSACTIONS)
    #     pred = load_csv_from_local(LOCAL_PREDICTIONS)
    
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['year_month'] = df['transaction_date'].dt.to_period('M')

    # predictions_output columns: user_id, predicted_risk, risk_label, credit_score, risk_tier
    return df, pred

# ─── Precompute ALL user features globally (correct normalization) ─────────────
@st.cache_data
def precompute_all_features(_df):
    """
    Engineer features for every user and normalise globally (across all users).
    This is the correct approach — per-user normalisation in the original code
    was wrong because MinMaxScaler fitted on a single row is meaningless.
    """
    columns = [
        'topup_frequency', 'topup_consistency', 'avg_topup_amount', 'topup_stability', 'topup_to_spend_ratio',
        'avg_monthly_spend', 'spend_stability', 'avg_payment_amount',
        'avg_monthly_txn_count', 'active_months_ratio', 'recency_days', 'txn_regularity',
        'unique_categories', 'category_entropy',
        'failed_rate', 'success_rate',
        'account_age', 'is_spend_only', 'raw_txn_count'
    ]

    ref_date = _df['transaction_date'].max() + pd.Timedelta(days=1)

    def engineer_pillars(group):
        payments = group[group['transaction_type'] == 'payment']
        topups   = group[group['transaction_type'] == 'topup']
        months_active = group['year_month'].nunique()
        date_range = (
            group['transaction_date'].max().to_period('M')
            - group['transaction_date'].min().to_period('M')
        ).n + 1

        # Pillar 1: Top-up Consistency
        f_topup_freq        = len(topups) / months_active if months_active > 0 else 0
        f_topup_consistency = topups['year_month'].nunique() / date_range
        f_avg_topup_amt     = topups['product_amount'].mean() if not topups.empty else 0
        if not topups.empty and topups['year_month'].nunique() > 1:
            topup_monthly   = topups.groupby('year_month')['product_amount'].sum()
            cv              = topup_monthly.std() / (topup_monthly.mean() + 1e-9)
            f_topup_stability = 1 / (1 + cv)
        else:
            f_topup_stability = 0
        f_topup_to_spend = topups['product_amount'].sum() / (payments['product_amount'].sum() + 1e-9)

        # Pillar 2: Spending Behaviour
        f_avg_monthly_spend = payments['product_amount'].sum() / months_active if months_active > 0 else 0
        if not payments.empty and payments['year_month'].nunique() > 1:
            spend_monthly   = payments.groupby('year_month')['product_amount'].sum()
            cv_s            = spend_monthly.std() / (spend_monthly.mean() + 1e-9)
            f_spend_stability = 1 / (1 + cv_s)
        else:
            f_spend_stability = 0
        f_avg_payment_amt = payments['product_amount'].mean() if not payments.empty else 0

        # Pillar 3: Transaction Frequency
        f_avg_monthly_txn  = len(payments) / months_active if months_active > 0 else 0
        f_active_ratio     = months_active / date_range
        f_recency          = (ref_date - group['transaction_date'].max()).days
        f_txn_regularity   = (
            payments.groupby('year_month').size().std()
            if payments['year_month'].nunique() > 1 else 0
        )

        # Pillar 4: Merchant Diversity
        f_unique_cats  = group['product_category'].nunique()
        cat_dist       = group['product_category'].value_counts(normalize=True)
        f_cat_entropy  = entropy(cat_dist)

        # Pillar 5: Reliability & Risk
        f_failed_rate  = (group['transaction_status'] == 'Failed').mean()
        f_success_rate = (group['transaction_status'] == 'Successful').mean()

        # Pillar 6: History Maturity
        f_account_age  = date_range
        is_spend_only  = 1 if len(topups) == 0 else 0
        raw_txn_count  = len(group)

        return pd.Series([
            f_topup_freq, f_topup_consistency, f_avg_topup_amt, f_topup_stability, f_topup_to_spend,
            f_avg_monthly_spend, f_spend_stability, f_avg_payment_amt,
            f_avg_monthly_txn, f_active_ratio, f_recency, f_txn_regularity,
            f_unique_cats, f_cat_entropy,
            f_failed_rate, f_success_rate,
            f_account_age, is_spend_only, raw_txn_count
        ])

    uf = _df.groupby('user_id').apply(engineer_pillars).reset_index()
    uf.columns = ['user_id'] + columns

    # ── Global normalisation (fitted across ALL users) ──
    scaler = MinMaxScaler()
    feat_to_scale = [c for c in columns if c not in ['is_spend_only', 'raw_txn_count']]
    uf_norm = uf.copy()
    uf_norm[feat_to_scale] = scaler.fit_transform(uf[feat_to_scale])

    # Invert signals where lower raw value = better
    for col in ['recency_days', 'txn_regularity', 'failed_rate']:
        uf_norm[col] = 1 - uf_norm[col]

    # ── Pillar sub-scores (0–1) ──
    uf['score_topup']       = uf_norm[['topup_frequency', 'topup_consistency',
                                        'avg_topup_amount', 'topup_stability',
                                        'topup_to_spend_ratio']].mean(axis=1)
    uf['score_spending']    = uf_norm[['avg_monthly_spend', 'spend_stability',
                                        'avg_payment_amount']].mean(axis=1)
    uf['score_frequency']   = uf_norm[['avg_monthly_txn_count', 'active_months_ratio',
                                        'recency_days', 'txn_regularity']].mean(axis=1)
    uf['score_diversity']   = uf_norm[['unique_categories', 'category_entropy']].mean(axis=1)
    uf['score_reliability'] = uf_norm[['failed_rate', 'success_rate']].mean(axis=1)
    uf['score_maturity']    = uf_norm['account_age']

    return uf.set_index('user_id')

df, pred_df = load_data()
all_features = precompute_all_features(df)

# ─── Lookup helpers ───────────────────────────────────────────────────────────
ICON_MAP = {
    'Food Delivery':      '🍕',
    'Grocery Shopping':   '🛒',
    'Online Shopping':    '🛍️',
    'Bus Ticket':         '🚌',
    'Taxi Fare':          '🚕',
    'Hotel Booking':      '🏨',
    'Flight Booking':     '✈️',
    'Electricity Bill':   '⚡',
    'Gas Bill':           '🔥',
    'Water Bill':         '💧',
    'Internet Bill':      '🌐',
    'Mobile Recharge':    '📱',
    'Streaming Service':  '📺',
    'Gaming Credits':     '🎮',
    'Insurance Premium':  '🛡️',
    'Loan Repayment':     '💰',
    'Rent Payment':       '🏠',
    'Wallet Top-up':      '💳',
    'Movie Ticket':       '🎬',  # ← was missing
    'Education Fee':      '📚',  # ← was missing
}

def get_icon(category):
    return ICON_MAP.get(category, '💳')


def score_label(risk_tier: str):
    """
    Map the XGBoost model's risk_tier to a display label and colour.
    risk_tier values in dataset: 'Poor', 'Fair', 'Good', 'Very Good'
    """
    mapping = {
        'Very Good': ('VERY GOOD', '#00d4a0'),
        'Good':      ('GOOD',      '#5b8ef0'),
        'Fair':      ('FAIR',      '#f59e0b'),
        'Poor':      ('NEEDS WORK','#ff5c5c'),
    }
    return mapping.get(risk_tier, ('FAIR', '#f59e0b'))


def get_user_data(user_id: str):
    """
    Build the full user data dict for the dashboard.
    Credit score and risk tier come from predictions_output (XGBoost model).
    Pillar sub-scores come from the globally-normalised feature table.
    """
    user_txn = df[df['user_id'] == user_id]
    if user_txn.empty:
        return None

    # ── Model predictions ──
    pred_row = pred_df[pred_df['user_id'] == user_id]
    if pred_row.empty:
        return None

    pred_row    = pred_row.iloc[0]
    credit_score = int(pred_row['credit_score'])
    risk_tier    = pred_row['risk_tier']
    risk_label_str = pred_row['risk_label']  # "Low Risk" / "High Risk"
    predicted_risk = int(pred_row['predicted_risk'])

    # ── Pillar sub-scores (0–1)
    feat = all_features.loc[user_id] if user_id in all_features.index else None
    pred_score_keys = [
        'score_spending', 'score_frequency', 'score_diversity',
        'score_reliability', 'score_maturity', 'score_topup'
    ]
    pred_scores = {k: float(pred_row[k]) for k in pred_score_keys if k in pred_row}
    if pred_scores:
        if feat is None:
            feat = pd.Series(pred_scores)
        else:
            for key, value in pred_scores.items():
                feat[key] = value

    # ── Last 10 transactions ──
    recent_txn = user_txn.sort_values('transaction_date', ascending=False).head(10)
    transactions = []
    for _, row in recent_txn.iterrows():
        icon     = get_icon(row['product_category'])
        date_str = row['transaction_date'].strftime('%d %b %Y')
        amount   = row['product_amount']
        # Topups are inflows (+), payments are outflows (-)
        if row['transaction_type'] == 'payment':
            amount = -amount
        transactions.append((
            icon,
            row['merchant_name'],
            date_str,
            amount,
            row['transaction_type'].upper()
        ))

    # ── Loan eligibility — tiered by risk_tier ──
    loan_limits = {'Very Good': 15000, 'Good': 10000, 'Fair': 5000, 'Poor': 1000}
    loan_eligible = loan_limits.get(risk_tier, 5000)

    # ── Account age ──
    account_age_months = int(feat['account_age']) if feat is not None else 0

    return {
        'name':          f"User {user_id}",
        'id':            user_id,
        'score':         credit_score,
        'max_score':     850,
        'risk_tier':     risk_tier,
        'risk_label':    risk_label_str,
        'predicted_risk': predicted_risk,
        'loan_eligible': loan_eligible,
        'account_age':   f"{account_age_months}mo",
        'transactions':  transactions,
        'features':      feat,
    }


def get_score_breakdown(feat, risk_tier: str):
    """
    Build the SCORE_BREAKDOWN dict from real pillar sub-scores.
    Display each pillar out of 100 (percentage of possible max).
    """
    if feat is None:
        return {}
    is_spend_only = int(feat.get('is_spend_only', 0))
    score_definitions = [
        ('Spending Behaviour', 'score_spending', '#f59e0b'),
        ('Transaction Frequency', 'score_frequency', '#5b8ef0'),
        ('Merchant Diversity', 'score_diversity', '#a855f7'),
        ('Reliability', 'score_reliability', '#00d4a0'),
        ('Account Maturity', 'score_maturity', '#ec4899'),
    ]
    items = {}
    for label, key, color in score_definitions:
        if key in feat:
            items[label] = (round(float(feat[key]) * 100), 100, color)
    if 'score_topup' in feat and not is_spend_only:
        items['Top-up Consistency'] = (round(float(feat['score_topup']) * 100), 100, '#00d4a0')
    return items


# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
.stApp, .main {
    background-color: #0a0f1a !important;
}
section { background-color: #0a0f1a !important; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0a0f1a !important;
    color: #e8edf5 !important;
}
.main .block-container {
    max-width: 1200px !important;
    padding: 1.5rem 2rem 4rem !important;
    background: #0a0f1a !important;
    margin: 0 auto;
}
h1, h2, h3, h4 { color: #e8edf5 !important; }

.stTextInput > div > div > input {
    background-color: #0d1526 !important;
    color: #e8edf5 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input::placeholder { color: #4a5a72 !important; }
.stTextInput > label { color: #e8edf5 !important; font-size: 13px !important; }



div[data-testid="stHorizontalBlock"] button {
    background: #0d1526 !important;
    border: 1px solid #1e2d40 !important;
    color: #4a5a72 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    padding: 12px 8px !important;
    transition: all 0.2s !important;
    flex: 1 !important;
    height: 48px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #00d4a044 !important;
    color: #00d4a0 !important;
}

[data-testid="stMetric"] {
    background: #0d1a2e;
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="stMetric"] label { color: #4a5a72 !important; font-size: 10px !important; letter-spacing: 1.5px !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e8edf5 !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00d4a0, #00a8ff) !important;
    border-radius: 3px !important;
}
.stProgress > div > div > div {
    background: #1e2d40 !important;
    border-radius: 3px !important;
}

.stButton > button {
    background: #0d1a2e !important;
    color: #e8edf5 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 16px 20px !important;
    width: 100% !important;
    font-size: 12px !important;
    transition: all 0.2s !important;
    min-height: 44px !important;
}
.stButton > button:hover {
    background: #1a2540 !important;
    border-color: #00d4a044 !important;
    color: #00d4a0 !important;
}

.btn-outline > .stButton > button {
    background: transparent !important;
    color: #e8edf5 !important;
    border: 1px solid #2a3a50 !important;
}

.stSelectbox > div > div {
    background: #0d1526 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 12px !important;
    color: #e8edf5 !important;
}

hr { border-color: #1e2d40 !important; }

.stAlert {
    background-color: #0d1a2e !important;
    border: 1px solid #1e2d40 !important;
    color: #e8edf5 !important;
}
.st-emotion-cache-1lsdpco { background-color: #0d1a2e !important; }

.stMarkdown, .stText, p { color: #e8edf5 !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0a0f1a; }
::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2a3a50; }
</style>
""", unsafe_allow_html=True)

# ─── UI Helpers ────────────────────────────────────────────────────────────────
def ring_svg(score, max_score=850, size=150):
    r = 56; cx = cy = size // 2; circ = 2 * 3.14159 * r
    fill = min(score / max_score, 1.0) * circ
    # Derive colour from actual score value (not risk_tier string here)
    if score >= 660:   color = '#00d4a0'
    elif score >= 590: color = '#5b8ef0'
    elif score >= 520: color = '#f59e0b'
    else:              color = '#ff5c5c'
    return f"""
    <div style="display:flex;align-items:center;justify-content:center;margin:8px 0">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e2d40" stroke-width="10"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="{color}" stroke-width="10" stroke-linecap="round"
          stroke-dasharray="{fill:.1f} {circ:.1f}"
          transform="rotate(-220 {cx} {cy})"/>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle"
          font-family="Space Mono, monospace" font-size="28" font-weight="700" fill="#e8edf5">{score}</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle"
          font-family="DM Sans, sans-serif" font-size="10" fill="#4a5a72">out of {max_score}</text>
      </svg>
    </div>"""

def card(html):
    return f'<div style="background:#0d1a2e;border:1px solid #1e2d40;border-radius:16px;padding:18px 16px;margin-bottom:14px">{html}</div>'

def label_html(t):
    return f'<div style="font-size:9px;letter-spacing:1.5px;color:#4a5a72;font-weight:600;margin-bottom:6px">{t}</div>'

def badge(t, c="#00d4a0", bg="#0e3d30", b="#00d4a044"):
    return f'<span style="background:{bg};color:{c};font-size:9px;font-weight:700;letter-spacing:1px;padding:3px 10px;border-radius:20px;border:1px solid {b}">{t}</span>'

def risk_badge(risk_label, predicted_risk):
    if predicted_risk == 0:
        return badge(f'✦ LOW RISK', '#00d4a0', '#0e3d30', '#00d4a044')
    else:
        return badge(f'⚠ HIGH RISK', '#ff5c5c', '#3d0e0e', '#ff5c5c44')

def pill(ic, l):
    return f'<span style="display:inline-flex;align-items:center;gap:5px;background:#1a2a40;border:1px solid #2a3a50;border-radius:20px;padding:4px 10px;font-size:10px;font-weight:600;letter-spacing:0.5px;margin-right:6px"><span style="color:{ic}">☁</span> {l}</span>'

def bar_html(label, val, max_val, color):
    pct = (val / max_val) * 100
    return f"""
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px">
        <span>{label}</span>
        <span style="color:{color};font-family:'Space Mono',monospace">{val}/{max_val}</span>
      </div>
      <div style="height:6px;background:#1e2d40;border-radius:3px;overflow:hidden">
        <div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:3px"></div>
      </div>
    </div>"""

def tx_row_html(icon, name, date, amount, tx_type):
    amt_color = '#00d4a0' if amount > 0 else '#ff5c5c'
    sign = '+' if amount > 0 else ''
    return f"""
    <div style="display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #1e2d4040">
      <div style="width:36px;height:36px;border-radius:10px;background:#1a2030;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0">{icon}</div>
      <div>
        <div style="font-size:13px;font-weight:500">{name}</div>
        <div style="font-size:11px;color:#4a5a72">{date}</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="color:{amt_color};font-size:13px;font-weight:600;font-family:'Space Mono',monospace">{sign}RM{abs(amount):.2f}</div>
        <div style="font-size:9px;letter-spacing:1px;color:#4a5a72">{tx_type}</div>
      </div>
    </div>"""

IMPROVEMENTS_MAP = {
    'Poor':      [
        "Increase top-up frequency — aim for at least 2× per month",
        "Reduce failed transactions by ensuring sufficient wallet balance",
        "Use at least 3 different merchant categories regularly",
        "Maintain consistent spending every month to build history",
    ],
    'Fair':      [
        "Top up consistently every month to boost your score",
        "Diversify merchants — try different spending categories",
        "Keep failed transaction rate below 10%",
        "Stay active for more consecutive months",
    ],
    'Good':      [
        "Top up regularly with stable amounts for a higher score",
        "Maintain weekly top-ups consistently",
        "Expand merchant diversity — aim for 5+ categories",
        "Keep transaction frequency above 5/month",
    ],
    'Very Good': [
        "Maintain your great top-up habits!",
        "Keep transaction frequency and merchant diversity high",
        "Ensure balance before transactions to avoid failures",
        "Stay consistently active every month",
    ],
}

# ─── Session State Initialization ────────────────────────────────────────────────
if 'tab' not in st.session_state:
    st.session_state.tab = 'Home'
if 'report_state' not in st.session_state:
    st.session_state.report_state = 'landing'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'USER_00001'
if 'ai_explanation' not in st.session_state:
    st.session_state.ai_explanation = None
if 'report_ai_explanation' not in st.session_state:
    st.session_state.report_ai_explanation = None

# ── LOGIN PAGE ──────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;margin-bottom:40px;padding:40px;background:#0d1a2e;border-radius:16px;border:1px solid #1e2d40">
          <div style="font-size:28px;font-weight:700;margin-bottom:6px;color:#e8edf5">GO Pinjam</div>
          <div style="font-size:12px;color:#4a5a72">Your Digital Credit Score · Powered by XGBoost</div>
        </div>
        """, unsafe_allow_html=True)

        # User ID selector for demo
        all_user_ids = sorted(pred_df['user_id'].unique().tolist())
        selected = st.selectbox('Select User ID (Demo)', all_user_ids, index=0)
        st.session_state.user_id = selected

        email    = st.text_input('Email', placeholder='ahmad@example.com',  key='login_email')
        password = st.text_input('Password', type='password', placeholder='••••••••', key='login_pass')
        st.markdown('<br>', unsafe_allow_html=True)

        if st.button('🔓 SIGN IN', use_container_width=True):
            if email and password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error('Please enter email and password')
    st.stop()

# ─── Load USER data ────────────────────────────────────────────────────────────
USER = get_user_data(st.session_state.user_id)

# Build the payload that will be sent to the AI explanation service.

def build_alibaba_payload(USER: dict) -> dict:
    feat = USER.get('features')

    # Spending stability from spending score pillar (0-1 → convert to 0-100)
    spending_stability = 0
    if feat is not None and 'score_spending' in feat:
        spending_stability = round(float(feat['score_spending']) * 100, 1)

    # Transactions per month from raw features
    transactions_per_month = 0
    if feat is not None and 'avg_monthly_txn_count' in feat:
        transactions_per_month = round(float(feat['avg_monthly_txn_count']), 1)

    # Merchant diversity from unique categories
    merchant_diversity = 0
    if feat is not None and 'unique_categories' in feat:
        merchant_diversity = int(feat['unique_categories'])

    # Top-up frequency
    topup_count = 0
    if feat is not None and 'topup_frequency' in feat:
        topup_count = round(float(feat['topup_frequency']), 1)

    # Refund count derived from failed rate
    refund_count = 0
    if feat is not None and 'failed_rate' in feat:
        raw_failed = float(feat.get('raw_txn_count', 1))
        refund_count = round(float(feat['failed_rate']) * raw_failed)

    return {
        'credit_score':       USER['score'],
        'risk_label':         USER['risk_label'],
        'risk_tier':          USER['risk_tier'],
        'transactions':       transactions_per_month,
        'merchant_diversity': merchant_diversity,
        'topup_count':        topup_count,
        'refund_count':       refund_count,
        'spending_stability': spending_stability,
    }

if USER is None:
    st.error(f"No data found for {st.session_state.user_id}. Please select another user.")
    st.stop()

SCORE_BREAKDOWN = get_score_breakdown(USER['features'], USER['risk_tier'])
TRANSACTIONS    = USER.get('transactions', [])
IMPROVEMENTS    = IMPROVEMENTS_MAP.get(USER['risk_tier'], IMPROVEMENTS_MAP['Good'])

# ─── Top bar ───────────────────────────────────────────────────────────────────
current_time = datetime.now().strftime('%H:%M')
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0 2px;margin-bottom:4px">
  <span style="font-size:13px;font-weight:600">{current_time}</span>
  <span style="font-size:11px;color:#4a5a72">XGBoost Model · {datetime.now().strftime('%d %b %Y')}</span>
</div>
""", unsafe_allow_html=True)

# ─── Nav Bar ───────────────────────────────────────────────────────────────────
nav_icons = {'Home': '🏠', 'Score': '📊', 'History': '📋', 'Report': '📄'}
cols = st.columns([0.9, 0.9, 0.9, 0.9, 1])
for i, (tab_name, icon) in enumerate(nav_icons.items()):
    with cols[i]:
        active = st.session_state.tab == tab_name
        label  = f'**{tab_name}**' if active else tab_name
        if st.button(label, key=f'nav_{tab_name}', use_container_width=True):
            st.session_state.tab = tab_name
            if tab_name == 'Report':
                st.session_state.report_state = 'landing'
            st.rerun()

with cols[4]:
    if st.button('Logout', key='logout_nav_btn', use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.tab = 'Home'
        st.session_state.report_state = 'landing'
        st.session_state.ai_explanation = None
        st.session_state.report_ai_explanation = None
        st.rerun()

active_colors = {'Home': '#00d4a0', 'Score': '#5b8ef0', 'History': '#f59e0b', 'Report': '#a855f7'}
act_color = active_colors[st.session_state.tab]
st.markdown(f'<div style="height:2px;background:{act_color};border-radius:2px;margin-bottom:18px"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB: HOME
# ══════════════════════════════════════════════════════════════════════
if st.session_state.tab == 'Home':

    s_label, s_color = score_label(USER['risk_tier'])

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px">
      <div>
        {label_html("WELCOME")}
        <div style="font-size:26px;font-weight:600;line-height:1.2">{USER['name']}</div>
        <div style="font-size:11px;color:#4a5a72;margin-top:3px">{USER['id']}</div>
      </div>
      <div style="width:50px;height:50px;border-radius:50%;background:#0e2a40;border:2px solid #00d4a044;display:flex;align-items:center;justify-content:center;font-size:22px">👤</div>
    </div>
    """, unsafe_allow_html=True)

    # Score card
    score_card_html = f"""{ring_svg(USER['score'], USER['max_score'], size=130)}
    <div style="flex:1;min-width:200px">
      {label_html('LOAN ELIGIBILITY')}
      <div style="color:#00d4a0;font-size:22px;font-weight:700;font-family:\'Space Mono\',monospace">RM{USER['loan_eligible']:,.0f}</div>
      {label_html('RISK ASSESSMENT')}<div style="margin-top:4px">{risk_badge(USER['risk_label'], USER['predicted_risk'])}</div>
    </div>
    <div style="width:100%;text-align:center;margin-top:10px">{badge(f'✦ {s_label}', s_color)}</div>"""
    st.markdown(card(score_card_html), unsafe_allow_html=True)

    # Quick actions
    st.markdown(label_html('QUICK ACTIONS'), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button('📄\nGenerate Report', key='qa_report', use_container_width=True):
            st.session_state.tab = 'Report'
            st.session_state.report_state = 'landing'
            st.rerun()
    with c2:
        if st.button('📊\nView Score', key='qa_score', use_container_width=True):
            st.session_state.tab = 'Score'
            st.rerun()
    with c3:
        if st.button('📋\nTransactions', key='qa_hist', use_container_width=True):
            st.session_state.tab = 'History'
            st.rerun()
    with c4:
        if st.button('🏦\nApply Loan', key='qa_loan', use_container_width=True):
            st.info(f"Eligible for up to RM{USER['loan_eligible']:,.0f} — Redirecting to CashLoan…")

    # Recent transactions
    col_recent, col_seeall = st.columns([0.6, 0.4])
    with col_recent:
        st.markdown(label_html('RECENT TRANSACTIONS'), unsafe_allow_html=True)
    with col_seeall:
        if st.button('SEE ALL →', key='see_all_btn', use_container_width=True):
            st.session_state.tab = 'History'
            st.rerun()

    if TRANSACTIONS:
        recent_html = ''.join(tx_row_html(*t) for t in TRANSACTIONS[:3])
        st.markdown(card(f'<div style="padding:0 2px">{recent_html}</div>'), unsafe_allow_html=True)
    else:
        st.markdown(card('<div style="color:#4a5a72;font-size:13px;text-align:center;padding:20px">No transactions found for this user.</div>'), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB: SCORE
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.tab == 'Score':

    st.markdown("""
    <div style="font-size:11px;color:#4a5a72;margin-bottom:2px">Calculated from all active transactions</div>
    <div style="font-size:22px;font-weight:600;margin-bottom:16px">Credit Score</div>
    """, unsafe_allow_html=True)

    s_label, s_color = score_label(USER['risk_tier'])

    col_ring, col_info = st.columns([1, 1])
    with col_ring:
        st.markdown(ring_svg(USER['score'], USER['max_score'], size=150), unsafe_allow_html=True)
    with col_info:
        feat = USER['features']
        txn_count = int(feat['raw_txn_count']) if feat is not None else 0
        active_months = int(feat['account_age']) if feat is not None else 0
        st.markdown(card(f"""
          <div style="font-size:9px;letter-spacing:1.5px;color:{s_color};margin-bottom:6px">{s_label}</div>
          <div style="font-size:11px;color:#a0b0c0;line-height:1.9">
            Risk: <span style="color:#e8edf5;font-weight:600">{USER['risk_label']}</span><br>
            Total Transactions: <span style="color:#e8edf5;font-weight:600">{txn_count}</span><br>
            Active Months: <span style="color:#e8edf5;font-weight:600">{active_months}</span><br>
            Loan Eligible: <span style="color:#00d4a0;font-weight:600">RM{USER['loan_eligible']:,.0f}</span>
          </div>
        """), unsafe_allow_html=True)

    st.markdown(label_html('MODEL SUMMARY'), unsafe_allow_html=True)
    st.markdown(card(f"""
      <div style="font-size:11px;color:#a0b0c0;line-height:1.8">
        Credit score is calculated using the user’s pillar sub-scores and the trained XGBoost risk model.
        This view reflects the score derived from transaction behaviour rather than simulated cloud score values.
      </div>
    """), unsafe_allow_html=True)

    # Score breakdown — computed from real pillar sub-scores
    st.markdown(label_html('SCORE BREAKDOWN (PILLAR SUB-SCORES)'), unsafe_allow_html=True)
    if SCORE_BREAKDOWN:
        bars_html = ''.join(bar_html(k, v, mx, c) for k, (v, mx, c) in SCORE_BREAKDOWN.items())
        st.markdown(card(bars_html), unsafe_allow_html=True)
    else:
        st.markdown(card('<div style="color:#4a5a72;font-size:13px">No pillar data available.</div>'), unsafe_allow_html=True)

    # Improvement tips — tailored to risk tier
    st.markdown(label_html('HOW TO IMPROVE'), unsafe_allow_html=True)
    tips_html = ''.join(f'<div style="font-size:12px;color:#a0b0c0;padding:5px 0">→ {t}</div>' for t in IMPROVEMENTS)
    st.markdown(card(tips_html), unsafe_allow_html=True)

    # Show the pillar sub-score chart for better visual insight.
    score_chart_df = pd.DataFrame(
        {label: [float(v)] for label, (v, mx, c) in SCORE_BREAKDOWN.items()}
    ).T.rename(columns={0: 'score'})
    score_chart_df.index.name = 'Pillar'
    if not score_chart_df.empty:
        st.markdown(label_html('PILLAR SCORES'), unsafe_allow_html=True)
        st.bar_chart(score_chart_df['score'], height=320)

    # ─── AI Explanation ────────────────────────────────────────────────────
    st.markdown(label_html('AI EXPLANATION (Alibaba Qwen)'), unsafe_allow_html=True)

    # Button to trigger AI explanation
    if st.button('🤖 Generate AI Explanation', key='ai_explain_score'):
        with st.spinner('Analyzing profile with Alibaba AI...'):
            payload = build_alibaba_payload(USER)
            explanation = get_explanation(payload)
            st.session_state.ai_explanation = explanation

    # Display explanation if generated
    if st.session_state.ai_explanation is not None:
        st.markdown(card(f"""
            <div style="font-size:9px;letter-spacing:1.5px;color:#a855f7;margin-bottom:10px">
                ☁ POWERED BY ALIBABA QWEN AI
            </div>
            <div style="font-size:12px;color:#a0b0c0;line-height:1.9;white-space:pre-wrap">
                {st.session_state.ai_explanation}
            </div>
        """), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB: HISTORY
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.tab == 'History':

    st.markdown('<div style="font-size:22px;font-weight:600;margin-bottom:2px">Transaction History</div>', unsafe_allow_html=True)

    user_txn = df[df['user_id'] == st.session_state.user_id].sort_values('transaction_date', ascending=False)
    if user_txn.empty:
        st.info('No transactions found for this user.')
    else:
        first_date = user_txn['transaction_date'].min().strftime('%b %Y')
        last_date  = user_txn['transaction_date'].max().strftime('%b %Y')
        st.markdown(f'<div style="font-size:12px;color:#4a5a72;margin-bottom:16px">{first_date} – {last_date}</div>', unsafe_allow_html=True)

        # Summary cards — based on actual data
        topup_txn   = user_txn[user_txn['transaction_type'] == 'topup']
        payment_txn = user_txn[user_txn['transaction_type'] == 'payment']
        total_topup = topup_txn['product_amount'].sum()
        total_spent = payment_txn['product_amount'].sum()

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div style="background:#0e2a1e;border:1px solid #1a4030;border-radius:14px;padding:16px">
              {label_html("TOTAL TOP-UP")}
              <div style="color:#00d4a0;font-size:18px;font-weight:700;font-family:'Space Mono',monospace">RM{total_topup:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div style="background:#2a0e0e;border:1px solid #401a1a;border-radius:14px;padding:16px">
              {label_html("TOTAL SPENT")}
              <div style="color:#ff5c5c;font-size:18px;font-weight:700;font-family:'Space Mono',monospace">RM{total_spent:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div style="background:#0e1a2e;border:1px solid #1a2a40;border-radius:14px;padding:16px">
              {label_html("TOTAL TXN")}
              <div style="color:#5b8ef0;font-size:18px;font-weight:700;font-family:'Space Mono',monospace">{len(user_txn)}</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            failed_count = (user_txn['transaction_status'] == 'Failed').sum()
            st.markdown(f"""
            <div style="background:#2a1e0e;border:1px solid #402e1a;border-radius:14px;padding:16px">
              {label_html("FAILED TXN")}
              <div style="color:#f59e0b;font-size:18px;font-weight:700;font-family:'Space Mono',monospace">{failed_count}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # Merchant diversity info
        merchant_count  = payment_txn['merchant_name'].nunique()
        category_count  = user_txn['product_category'].nunique()
        diversity_label = 'HIGH' if category_count >= 4 else ('MEDIUM' if category_count >= 2 else 'LOW')
        div_color       = '#00d4a0' if category_count >= 4 else ('#f59e0b' if category_count >= 2 else '#ff5c5c')
        st.markdown(f"""
        <div style="background:#1a2540;border:1px solid #2a3a60;border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center">
          <div>
            <div style="font-size:12px;font-weight:600;letter-spacing:0.5px">🏪 {merchant_count} MERCHANTS · {category_count} CATEGORIES</div>
            <div style="font-size:11px;color:#4a5a72;margin-top:3px">Merchant diversity: {diversity_label}</div>
          </div>
          <span style="color:{div_color};font-size:12px;font-weight:700">{diversity_label} ↑</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(label_html('ALL TRANSACTIONS'), unsafe_allow_html=True)
        all_tx = ''.join(tx_row_html(*t) for t in TRANSACTIONS)
        st.markdown(card(f'<div style="padding:0 2px">{all_tx}</div>'), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB: REPORT
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.tab == 'Report':

    st.markdown('<div style="font-size:22px;font-weight:600;margin-bottom:2px">FinPassport Report</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#4a5a72;margin-bottom:22px">Generate a digital credit document</div>', unsafe_allow_html=True)

    if st.session_state.report_state == 'landing':
        features_info = [
            ('🤖', '#1a1040', 'XGBoost AI-Powered', 'Score calculated from pillar sub-scores and the trained risk model.'),
        ]
        for icon, bg, title, desc in features_info:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:14px;background:#0d1a2e;border:1px solid #1e2d40;border-radius:14px;padding:16px;margin-bottom:10px">
              <div style="width:40px;height:40px;border-radius:10px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">{icon}</div>
              <div>
                <div style="font-size:13px;font-weight:600;margin-bottom:4px">{title}</div>
                <div style="font-size:12px;color:#a0b0c0;line-height:1.6">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('GENERATE FINPASSPORT →', key='gen_btn'):
            st.session_state.report_state = 'generating'
            st.rerun()

    elif st.session_state.report_state == 'generating':
        st.markdown("""
        <div style="text-align:center;padding:30px 0 10px">
          <div style="font-size:48px;margin-bottom:12px">⏳</div>
          <div style="color:#00d4a0;font-size:13px;font-weight:600;letter-spacing:0.5px;margin-bottom:8px">Generating PDF report...</div>
        </div>""", unsafe_allow_html=True)

        bar_ph  = st.empty()
        pct_ph  = st.empty()
        stat_ph = st.empty()

        for pct in range(0, 101, random.randint(6, 14)):
            pct = min(pct, 100)
            bar_ph.progress(pct / 100)
            pct_ph.markdown(
                f'<div style="text-align:center;font-size:32px;font-weight:700;font-family:Space Mono,monospace">{pct}%</div>',
                unsafe_allow_html=True
            )
            phase = 'Processing' if pct < 50 else 'Validating' if pct < 85 else 'Finalising'
            stat_ph.markdown(f"""
            <div style="display:flex;gap:10px;justify-content:center;margin-top:6px">
              {pill('#00d4a0', phase)}
            </div>""", unsafe_allow_html=True)
            time.sleep(0.18)

        time.sleep(0.4)
        st.session_state.report_state = 'done'
        st.rerun()

    elif st.session_state.report_state == 'done':
        s_label, s_color = score_label(USER['risk_tier'])
        feat = USER['features']

        def mini_bar(label, val, color):
            pct = min(val, 100)
            return f"""
            <div style="display:flex;align-items:center;font-size:11px;margin-bottom:5px">
              <span style="width:72px;color:#a0b0c0">{label}</span>
              <div style="flex:1;height:5px;background:#1e2d40;border-radius:3px;overflow:hidden;margin:0 8px">
                <div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:3px"></div>
              </div>
              <span style="color:{color};font-family:'Space Mono',monospace;font-size:10px">{int(val)}</span>
            </div>"""

        bars3 = ''
        if SCORE_BREAKDOWN:
            for k, (v, mx, c) in list(SCORE_BREAKDOWN.items())[:3]:
                short = k.split()[0]
                bars3 += mini_bar(short, v, c)

        st.markdown(card(f"""
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
            <div>
              {label_html("TNG FINPASSPORT")}
              <div style="font-size:18px;font-weight:600">{USER['name']}</div>
              <div style="font-size:11px;color:#4a5a72">{USER['id']}</div>
            </div>
          </div>
          <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:14px">
            <div>
              {label_html("CREDIT SCORE")}
              <div style="font-size:38px;font-weight:700;font-family:'Space Mono',monospace;line-height:1;color:{s_color}">{USER['score']}</div>
              <div style="font-size:10px;color:#4a5a72">/ {USER['max_score']} • {s_label}</div>
            </div>
            <div style="flex:1;padding-top:6px">{bars3}</div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:12px;font-size:11px">
            <div>{label_html("LOAN ELIGIBLE")}<div style="color:#00d4a0;font-weight:600;font-family:'Space Mono',monospace">RM{USER['loan_eligible']:,.0f}</div></div>
            <div>{label_html("RISK LABEL")}<div style="font-weight:600">{USER['risk_label']}</div></div>
            <div>{label_html("ACCOUNT AGE")}<div style="font-weight:600">{USER['account_age']}</div></div>
            <div>{label_html("GENERATED")}<div style="font-weight:600">{datetime.now().strftime('%d %b %Y')}</div></div>
          </div>
        """), unsafe_allow_html=True)

        report_lines = [
            "TNG FINPASSPORT+ CREDIT REPORT",
            "=" * 40,
            f"User ID    : {USER['id']}",
            f"Credit Score: {USER['score']} / {USER['max_score']}",
            f"Risk Tier  : {USER['risk_tier']}",
            f"Risk Label : {USER['risk_label']}",
            f"Loan Eligible: RM{USER['loan_eligible']:,.0f}",
            f"Account Age: {USER['account_age']}",
            f"Generated  : {datetime.now().strftime('%d %b %Y %H:%M')}",
            "",
            "PILLAR SUB-SCORES",
            "-" * 40,
        ]
        for k, (v, mx, _) in SCORE_BREAKDOWN.items():
            report_lines.append(f"{k:<26}: {v:>3}/{mx}")
        report_lines += [""]
        report_text = "\n".join(report_lines)

        st.download_button(
            label='⬇ DOWNLOAD REPORT (.txt)',
            data=report_text,
            file_name=f"FinPassport_{USER['id']}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime='text/plain',
        )

        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button('🏦 APPLY LOAN VIA CASHLOAN', key='loan_btn'):
            st.info(f"Redirecting to CashLoan with your FinPassport credential — eligible up to RM{USER['loan_eligible']:,.0f}…")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('↺ Regenerate', key='regen_btn'):
            st.session_state.report_state = 'landing'
            st.rerun()

    # ─── AI Explanation in Report ─────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(label_html('AI CREDIT ANALYSIS (Alibaba Qwen)'), unsafe_allow_html=True)

    if st.button('🤖 Get AI Analysis', key='ai_report_btn'):
        with st.spinner('Analyzing report with Alibaba AI...'):
            payload = build_alibaba_payload(USER)
            ai_text = get_explanation(payload)
            st.session_state.report_ai_explanation = ai_text

    if st.session_state.report_ai_explanation is not None:
        st.markdown(card(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                <span style="font-size:9px;letter-spacing:1.5px;color:#a855f7">
                    ☁ ALIBABA QWEN AI ANALYSIS
                </span>
            </div>
            <div style="font-size:12px;color:#a0b0c0;line-height:1.9;white-space:pre-wrap">
                {st.session_state.report_ai_explanation}
            </div>
        """), unsafe_allow_html=True)
