from openai import OpenAI

client = OpenAI(
    api_key="sk-d63f74d2178f4678b248874f176bafa3",
    base_url="https://ws-p52zzkssx1i29czz.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)

def get_explanation(user_data: dict) -> str:
    prompt = f"""
        You are analyzing a user using a CUSTOM fintech scoring system (NOT FICO).

        System Rules:
        Score range: 300–900
        Higher score = better behavior

        Scoring Factors (IMPORTANT — follow this strictly):
        - Top-up consistency (25%)
        - Spending behavior (20%)
        - Transaction frequency (20%)
        - Merchant diversity (15%)
        - Refund behavior (10%)
        - Account age (10%)

        Interpretation:
        - 700–900 → Excellent
        - 600–699 → Good
        - 500–599 → Fair
        - below 500 → High Risk

        User Data:
        Credit Score: {user_data['credit_score']}
        Risk Level: {user_data['risk_label']}
        Risk Tier: {user_data['risk_tier']}

        Behavior Data:
        - Transactions per month: {user_data['transactions']}
        - Merchant diversity: {user_data['merchant_diversity']}
        - Top-up frequency: {user_data['topup_count']}
        - Refund count: {user_data['refund_count']}
        - Spending stability: {user_data['spending_stability']}

        IMPORTANT:
        - ONLY explain based on the given data
        - DO NOT assume missing values
        - DO NOT mention external systems
        - Keep explanation SHORT, CLEAR, and USER-FRIENDLY

        Explain in English using a short markdown-style response:
        1. Why this score was assigned (2–3 lines)
        2. Key factors as bullet points
        3. Three actionable improvements as short bullets
        4. One sentence of encouragement
    """

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Alibaba AI is not available: {str(e)}"