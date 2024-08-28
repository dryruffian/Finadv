import streamlit as st
import logging
import base64
import plotly.graph_objects as go
from Config import Config
from data_processing import setup_data_and_vectorstore, create_qa_chain
from user_profile import save_user_profile, load_user_profile, calculate_risk_score
from investment_advice import get_investment_advice, create_risk_profile_chart, create_investment_allocation_chart
from financial_calculations import rebalance_portfolio
from stock_analysis import fetch_stock_data, predict_stock_price
from ui_components import render_header, render_custom_css

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Home"

    render_custom_css()
    render_header()

    tab1, tab2, tab3, tab4, tab6 = st.tabs(
        ["🏠 Home", "👤 Profile & Risk", "💡 Investment Advice", "📊 Financial Dashboard", "🎓 Financial Education Hub"])

    with tab1:
        st.markdown("## Welcome to MarketWealth Genius! 🚀")
        st.markdown("Your personal AI-powered financial advisor, here to guide you through your investment journey.")

        st.markdown("### 📘 How to Use MarketWealth Genius")
        for line in Config.HOW_TO_USE.split('\n'):
            if line.strip():
                st.markdown(f"- {line.strip()}")

        st.markdown("### 🌟 Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🤖 AI-Powered Advice")
            st.write("Get personalized investment recommendations based on your profile and market data.")
        with col2:
            st.markdown("#### 📊 Risk Assessment")
            st.write("Understand your risk tolerance and how it affects your investment strategy.")
        with col3:
            st.markdown("#### 🎓 Educational Resources")
            st.write("Access a wealth of information to improve your financial literacy.")

        st.markdown("### 🚀 Get Started")
        if st.button("Begin Your Investment Journey"):
            st.session_state.active_tab = "Profile & Risk"
            # st.experimental_rerun()

    with tab2:
        st.markdown("## 👤 User Profile and Risk Assessment")

        if 'profile' not in st.session_state:
            st.session_state.profile = {
                "age": "",
                "gender": "Male",
                "Avenue": "",
                "Purpose": "",
                "Duration": "",
                "risk_score": 0,
                "risk_assessment_answers": []
            }

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Personal Information")
            st.session_state.profile["age"] = st.number_input("Age", min_value=18, max_value=100,
                                                              value=int(st.session_state.profile["age"]) if
                                                              st.session_state.profile["age"] else 30)
            st.session_state.profile["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"],
                                                              index=["Male", "Female", "Other"].index(
                                                                  st.session_state.profile["gender"]))
            st.session_state.profile["Avenue"] = st.selectbox("Investment Avenue",
                                                              ["Stocks", "Bonds", "Real Estate", "Mutual Funds", "ETFs",
                                                               "Cryptocurrencies"],
                                                              index=0 if not st.session_state.profile["Avenue"] else [
                                                                  "Stocks", "Bonds", "Real Estate", "Mutual Funds",
                                                                  "ETFs", "Cryptocurrencies"].index(
                                                                  st.session_state.profile["Avenue"]))
            st.session_state.profile["Purpose"] = st.selectbox("Investment Purpose",
                                                               ["Retirement", "Short-term Goals", "Long-term Growth",
                                                                "Income Generation", "Capital Preservation"],
                                                               index=0 if not st.session_state.profile["Purpose"] else [
                                                                   "Retirement", "Short-term Goals", "Long-term Growth",
                                                                   "Income Generation", "Capital Preservation"].index(
                                                                   st.session_state.profile["Purpose"]))
            st.session_state.profile["Duration"] = st.slider("Investment Duration (years)", 1, 30,
                                                             value=int(st.session_state.profile["Duration"]) if
                                                             st.session_state.profile["Duration"] else 10)
            st.session_state.profile["income_stability"] = st.selectbox("Income Stability", ["Low", "Medium", "High"],
                                                                        index=1)
        with col2:
            st.sidebar.markdown("### Risk Assessment")
            user_answers = []
            for i, question in enumerate(Config.RISK_ASSESSMENT_QUESTIONS, 1):
                answer = st.sidebar.select_slider(question, options=['1', '2', '3', '4', '5'], key=f'question_{i}')
                user_answers.append(answer)

            if st.sidebar.button("Calculate Risk Profile"):
                try:
                    risk_score = calculate_risk_score(user_answers)
                    st.session_state.profile["risk_score"] = risk_score
                    st.session_state.profile["risk_assessment_answers"] = user_answers
                    st.sidebar.success(f"Your risk score: {risk_score[0]}")
                    st.sidebar.plotly_chart(create_risk_profile_chart(risk_score[0]), use_container_width=True)
                except ValueError as e:
                    st.sidebar.error(str(e))

            st.sidebar.header("We value your feedback! 🌟")

            feedback = st.sidebar.text_area("📝 Leave Feedback", "Share your thoughts and suggestions here...")
            rating = st.sidebar.slider("📊 Rate Your Experience", 1, 5, 3)
            suggestions = st.sidebar.text_input("🗣 Share Suggestions", "Any ideas for new features or improvements?")
            issue = st.sidebar.text_area("🧩 Report Issues", "Describe any problems you encountered...")

            if st.sidebar.button("Submit Feedback"):
                st.sidebar.success("Thank you for your feedback! 👍")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Profile"):
                save_user_profile(st.session_state.profile)
                st.success("Profile saved successfully!")
        with col2:
            if st.button("Load Profile"):
                loaded_profile = load_user_profile()
                if loaded_profile:
                    st.session_state.profile.update(loaded_profile)
                    st.success("Profile loaded successfully!")
                else:
                    st.error("No profile found.")

        st.markdown("## 🔄 Portfolio Rebalancing")

        st.markdown("### Current Portfolio Allocation")
        current_allocation = {}
        for asset in ['Stocks', 'Bonds', 'Real Estate', 'Cash']:
            current_allocation[asset] = st.number_input(f"Current {asset} allocation (%)", min_value=0, max_value=100,
                                                        value=25, key=f"current_{asset}")

        st.markdown("### Target Portfolio Allocation")
        target_allocation = {}
        for asset in ['Stocks', 'Bonds', 'Real Estate', 'Cash']:
            target_allocation[asset] = st.number_input(f"Target {asset} allocation (%)", min_value=0, max_value=100,
                                                       value=25, key=f"target_{asset}")

        if st.button("Rebalance Portfolio"):
            # Check if allocations sum to 100%
            if sum(current_allocation.values()) != 100 or sum(target_allocation.values()) != 100:
                st.error("Both current and target allocations must sum to 100%. Please adjust your inputs.")
            else:
                rebalancing_actions = rebalance_portfolio(current_allocation, target_allocation)
                st.markdown("### Rebalancing Actions")
                for asset, action in rebalancing_actions.items():
                    st.write(f"- {asset}: {action}")

                # Create a bar chart to visualize the rebalancing
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(current_allocation.keys()), y=list(current_allocation.values()),
                                     name='Current Allocation'))
                fig.add_trace(go.Bar(x=list(target_allocation.keys()), y=list(target_allocation.values()),
                                     name='Target Allocation'))
                fig.update_layout(title='Portfolio Allocation: Current vs Target', barmode='group',
                                  xaxis_title='Asset Class', yaxis_title='Allocation (%)')
                st.plotly_chart(fig, use_container_width=True)

                # Calculate the total portfolio value (assuming $100,000 for this example)
                total_portfolio_value = 100000
                st.markdown("### Rebalancing Transactions")
                st.markdown(f"Assuming a total portfolio value of ${total_portfolio_value:,}")
                for asset, action in rebalancing_actions.items():
                    action_type, percentage = action.split()
                    percentage = float(percentage.strip('%'))
                    amount = total_portfolio_value * (percentage / 100)
                    if action_type == "Buy":
                        st.write(f"- {asset}: Buy ${amount:,.2f}")
                    else:
                        st.write(f"- {asset}: Sell ${amount:,.2f}")

    with tab3:
        st.markdown("## 💡 Investment Advice")

        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None

        if st.button("Load Financial Data"):
            with st.spinner("Loading data..."):
                sklearn_store = setup_data_and_vectorstore()
                if sklearn_store:
                    st.session_state.qa_chain = create_qa_chain(sklearn_store)
                    st.success("Data loaded successfully. You can now ask for investment advice!")
                else:
                    st.error("Failed to load data.")

        question = st.text_area("What would you like to know about investing?", height=100)

        if st.button("Get Personalized Advice"):
            if st.session_state.qa_chain:
                with st.spinner("Generating your personalized investment advice..."):
                    advice = get_investment_advice(st.session_state.profile, question, st.session_state.qa_chain)
                    st.markdown("### 🎯 Your Personalized Investment Advice")
                    st.info(advice)

                    # Add FD rate information
                    fd_rates = Config.FD_RATES
                    st.markdown("#### Fixed Deposit Rates")
                    for bank, rate in fd_rates.items():
                        st.write(f"- {bank}: {rate:.2f}%")

                    st.plotly_chart(create_investment_allocation_chart(advice),
                                    use_container_width=True)
            else:
                st.error("Please load the financial data first by clicking the 'Load Financial Data' button.")

    with tab4:
        st.markdown("## 📊 Financial Dashboard")

        # Mock financial data
        savings = 15000
        investments = 50000
        debt = 5000

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Savings", f"${savings:,}", "+5%")

        with col2:
            st.metric("Investments", f"${investments:,}", "+12%")

        with col3:
            st.metric("Debt", f"${debt:,}", "-10%")

        # Interactive chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May'], y=[10000, 12000, 11000, 15000, 16000], name='Savings'))
        fig.add_trace(go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May'], y=[40000, 42000, 45000, 48000, 50000],
                                 name='Investments'))
        fig.update_layout(title='Financial Growth Over Time', xaxis_title='Month', yaxis_title='Amount ($)')
        st.plotly_chart(fig, use_container_width=True)

        # Stock prediction feature
        st.markdown("### 📈 Stock Price Prediction")
        stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL for Apple):")
        if stock_symbol:
            data = fetch_stock_data(stock_symbol)
            if data is not None:
                st.line_chart(data['Close'])
                future_dates, predictions = predict_stock_price(data)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical'))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Predicted'))
                fig.update_layout(title=f'{stock_symbol} Stock Price Prediction', xaxis_title='Date',
                                  yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.markdown("## 🎓 Financial Education Hub")
        st.markdown("### 📚 Educational Resources")
        with st.expander("📘 Investment Basics"):
            st.write(
                "Learn the fundamental concepts of investing, including asset classes, risk vs. return, "
                "and diversification.")
            st.markdown('[Learn more about Investment Basics](https://www.investopedia.com/terms/i/investment.asp)')

        with st.expander("🛡 Risk Management Strategies"):
            st.write(
                "Discover techniques to manage and mitigate investment risks, including portfolio diversification and "
                "hedging strategies.")
            st.markdown(
                '[Learn more about Risk Management Strategies]('
                'https://www.sailpoint.com/identity-library/what-is-risk-management-strategy/)')

        with st.expander("💰 Tax-Efficient Investing"):
            st.write("Explore strategies to minimize your tax liability while maximizing your investment returns.")
            st.markdown(
                '[Learn more about Tax-Efficient Investing]('
                'https://www.financialexpress.com/money/income-tax-tax-efficient-investment-planning-a-practical-guide-to'
                '-wealth-building-for-indian-investors-3384958/)')

        with st.expander("📅 Retirement Planning"):
            st.write(
                "Learn how to plan and save effectively for your retirement, including information on 401(k)s, IRAs, "
                "and other retirement accounts.")
            st.markdown(
                '[Learn more about Retirement Planning](https://www.investopedia.com/terms/r/retirement-planning.asp)')

        with st.expander("📈 Market Analysis Techniques"):
            st.write(
                "Discover various methods for analyzing financial markets, including fundamental and technical analysis.")
            st.markdown(
                '[Learn more about Market Analysis Techniques]('
                'https://nwokediothniel.medium.com/understanding-market-analysis-techniques-a-comprehensive-guide'
                '-928a124a2e7b)')

        st.markdown("### 🤔 Sample Questions to Get You Started")
        for category, questions in Config.SAMPLE_QUESTIONS.items():
            with st.expander(category):
                for q in questions:
                    st.write(f"• {q}")

        # Interactive Learning Modules
        st.markdown("### 🧠 Interactive Learning Modules")
        module = st.selectbox("Choose a learning module:",
                              ["Investment Strategies", "Risk Assessment", "Financial Planning"])

        if module == "Investment Strategies":
            st.write("This module covers various investment strategies.")
            strategy = st.radio("Select a strategy to learn more:",
                                ["Value Investing", "Growth Investing", "Index Investing"])
            if strategy:
                st.write(f"You selected {strategy}. Here's a brief overview...")
        elif module == "Risk Assessment":
            st.write("Learn how to assess your risk tolerance.")
            risk_score = st.slider("Rate your risk tolerance:", 1, 10, 5)
            st.write(
                f"Based on your score of {risk_score}, your risk tolerance is {'Low' if risk_score <= 3 else 'Medium' if risk_score <= 7 else 'High'}.")
        elif module == "Financial Planning":
            st.write("Create a basic financial plan.")
            income = st.number_input("Enter your monthly income:", min_value=0)
            expenses = st.number_input("Enter your monthly expenses:", min_value=0)
            if income and expenses:
                savings = income - expenses
                st.write(f"Your monthly savings potential is: ${savings}")

        # Financial Calculator
        st.markdown("### 🧮 Financial Calculator")
        calc_type = st.selectbox("Choose a calculator:", ["Compound Interest", "Loan Repayment", "Retirement Savings"])

        if calc_type == "Compound Interest":
            principal = st.number_input("Initial investment:", min_value=0.0)
            rate = st.number_input("Annual interest rate (%):", min_value=0.0, max_value=100.0)
            time = st.number_input("Time period (years):", min_value=0)
            if st.button("Calculate Compound Interest"):
                result = principal * (1 + rate / 100) ** time
                st.write(f"After {time} years, your investment will grow to: ${result:.2f}")
        elif calc_type == "Loan Repayment":
            loan_amount = st.number_input("Loan amount:", min_value=0.0)
            loan_term = st.number_input("Loan term (years):", min_value=0)
            interest_rate = st.number_input("Annual interest rate (%):", min_value=0.0, max_value=100.0)
            if st.button("Calculate Monthly Payment"):
                monthly_rate = interest_rate / (12 * 100)
                months = loan_term * 12
                payment = loan_amount * (monthly_rate * (1 + monthly_rate) * months) / ((1 + monthly_rate) * months - 1)
                st.write(f"Your monthly payment will be: ${payment:.2f}")
        elif calc_type == "Retirement Savings":
            current_age = st.number_input("Current age:", min_value=0, max_value=100)
            retirement_age = st.number_input("Retirement age:", min_value=current_age, max_value=100)
            monthly_contribution = st.number_input("Monthly contribution:", min_value=0.0)
            annual_return = st.number_input("Expected annual return (%):", min_value=0.0, max_value=100.0)
            if st.button("Calculate Retirement Savings"):
                months = (retirement_age - current_age) * 12
                total_savings = monthly_contribution * ((1 + annual_return / 1200)(months) - 1) / (annual_return / 1200)
                st.write(f"At retirement, your savings could grow to: ${total_savings:.2f}")

        # Glossary of Financial Terms
        st.markdown("### 📖 Glossary of Financial Terms")
        terms = {
            "Asset Allocation": "The process of dividing investments among different kinds of assets, such as stocks, "
                                "bonds, and cash, to optimize the risk/reward tradeoff based on an individual's specific "
                                "situation and goals.",
            "Diversification": "A risk management strategy that mixes a wide variety of investments within a portfolio to "
                               "potentially limit exposure to any single asset or risk.",
            "Compound Interest": "Interest calculated on the initial principal and also on the accumulated interest of "
                                 "previous periods of a deposit or loan.",
            "Dollar-Cost Averaging": "An investment strategy in which an investor divides up the total amount to be "
                                     "invested across periodic purchases of a target asset in an effort to reduce the "
                                     "impact of volatility on the overall purchase.",
            "Exchange-Traded Fund (ETF)": "A type of security that tracks an index, sector, commodity, or other asset, "
                                          "but which can be purchased or sold on a stock exchange the same as a regular "
                                          "stock."
        }
        term = st.selectbox("Select a term to learn more:", list(terms.keys()))
        st.write(terms[term])

        # Financial News Feed
        st.markdown("### 📰 Latest Financial News")
        # In a real application, you would fetch this data from a financial news API
        news_items = [
            "Stock Market Reaches New High",
            "Federal Reserve Announces Interest Rate Decision",
            "Tech Stocks Rally on Positive Earnings Reports",
            "Global Economic Outlook Improves",
            "New Cryptocurrency Regulations Proposed"
        ]
        for item in news_items:
            st.write(f"• {item}")

        st.markdown("---")
        st.markdown('<p style="text-align: center;">Created with ❤ using Streamlit and LangChain</p>',
                    unsafe_allow_html=True)

        # Chatbot interface
        st.markdown("### 💬 Chat with MarketWealth Genius")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about investing!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if st.session_state.qa_chain:
                    message_placeholder = st.empty()
                    full_response = ""
                    for response in st.session_state.qa_chain.run(prompt):
                        full_response += response
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    st.markdown(
                        "Please load the financial data first by clicking the 'Load Financial Data' button in the "
                        "Investment Advice tab.")

            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    logging.info("Starting MarketWealth Genius: Your AI Financial Advisor")
    main()
