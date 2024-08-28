class Config:
    DATA_FILE = 'Finance_data.csv'
    HOW_TO_USE = """
    ⬇ Scroll down and click 'BEGIN YOUR INVESTMENT JOURNEY'.
    📊 Click 'Load Data' to initialize the AI.
    📝 Complete the risk assessment questionnaire.
    👤 Fill in your profile information.
    ❓ Enter your investment query in the text area.
    🚀 Click 'Get Advice' to receive personalized investment recommendations.
    📈 Review the advice and investment allocation chart.
    🔄 Use the portfolio rebalancing tool if needed.
    📚 Explore educational resources for more information.
    """
    SAMPLE_QUESTIONS = {
        "Retirement 👴👵": [
            "What's a good investment strategy for retirement in my 30s?",
            "How should I adjust my retirement portfolio as I get closer to retirement age?"
        ],
        "Short-term Goals 🏠💍": [
            "How should I invest for a down payment on a house in 5 years?",
            "What are good investment options for saving for a wedding in 2 years?"
        ],
        "Long-term Growth 📈💰": [
            "What's a good strategy for long-term wealth building?",
            "How can I create a diversified portfolio for maximum growth over 20 years?"
        ],
        "Low-risk Options 🛡💸": [
            "What are some low-risk investment options for beginners?",
            "How can I protect my savings from inflation with minimal risk?"
        ],
        "Tax-efficient Investing 📑💱": [
            "What are the best options for tax-efficient investing?",
            "How can I minimize my tax liability while maximizing returns?"
        ]
    }
    RISK_ASSESSMENT_QUESTIONS = [
        "On a scale of 1 to 5, how comfortable are you with taking risks in your investments? 😰😐😎",
        "How would you react if your investment lost 10% of its value in a month? 😱😕🤔",
        "How long do you plan to hold your investments before needing to access the funds? ⏱💼",
        "What is your primary goal for investing? 🎯💸"
    ]

    FD_RATES = {
        "Unity Small Finance Bank": 9.0,
        "Utkarsh Small Finance Bank": 8.5,
        "RBL Bank": 8.1,
        "SBM Bank India": 8.25,
        "Bandhan Bank": 8.0
    }
    FD_SENIOR_CITIZEN_RATE_PREMIUM = 0.5

    openai_api_key = ("sk-proj-nToX3E7wOgNwqbMy18noiRUc5mn2Du2bwNgvEnAhoI1Y4YB9cH-tS8fzqI1x7MmAdpgqS_yhUBT3BlbkFJ7E"
                      "-S2hED5mkdZoGm3KPGf1v6Gr-MODjG8eBCbKxhZ4ZHtUEat4wsuHm-V5JbKqjjhYEepKlZkA")