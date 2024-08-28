import plotly.graph_objects as go
import plotly.express as px
from financial_calculations import get_asset_allocation, monte_carlo_simulation
import numpy as np
from user_profile import calculate_risk_score, get_risk_tolerance

def get_investment_advice(profile, question, qa_chain):
    if 'risk_assessment_answers' in profile and profile['risk_assessment_answers']:
        risk_score, risk_profile = calculate_risk_score(profile['risk_assessment_answers'])
    else:
        risk_score, risk_profile = 3, "Moderate"  # Default values if no assessment done

    risk_tolerance = get_risk_tolerance(profile['age'], profile['Duration'], profile.get('income_stability', 'Medium'))

    expected_returns = np.array([0.1, 0.05, 0.07, 0.02])  # Example returns for Stocks, Bonds, Real Estate, Cash
    cov_matrix = np.array([[0.04, -0.02, 0.01, 0],
                           [-0.02, 0.01, -0.005, 0],
                           [0.01, -0.005, 0.02, 0],
                           [0, 0, 0, 0.001]])
    asset_allocation = get_asset_allocation(risk_profile, expected_returns, cov_matrix)

    initial_investment = profile.get('initial_investment', 10000)
    monthly_contribution = profile.get('monthly_contribution', 500)
    projection_years = profile['Duration']
    projection_results = monte_carlo_simulation(initial_investment, monthly_contribution, projection_years)

    context = f"""
    Risk Profile: {risk_profile} (Score: {risk_score:.2f})
    Risk Tolerance: {risk_tolerance:.2f}

    Recommended Asset Allocation:
    {', '.join(f'{asset}: {allocation:.2%}' for asset, allocation in asset_allocation.items())}

    Financial Projection (in {projection_years} years):
    - Pessimistic scenario: ${projection_results[0]:,.2f}
    - Most likely scenario: ${projection_results[1]:,.2f}
    - Optimistic scenario: ${projection_results[2]:,.2f}

    Based on this information, provide tailored investment advice for the following question:
    {question}
    """

    response = qa_chain({"query": context})
    return response["result"]

def create_risk_profile_chart(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Profile", 'font': {'size': 24, 'color': '#E0E0E0'}},
        gauge={
            'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "#E0E0E0"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#E0E0E0",
            'steps': [
                {'range': [1, 2], 'color': '#10B981'},
                {'range': [2, 3], 'color': '#3B82F6'},
                {'range': [3, 4], 'color': '#F59E0B'},
                {'range': [4, 5], 'color': '#EF4444'}],
            'threshold': {
                'line': {'color': "#E0E0E0", 'width': 4},
                'thickness': 0.75,
                'value': risk_score}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#E0E0E0", 'family': "Poppins"})
    return fig

def create_investment_allocation_chart(advice):
    allocations = {
        'Stocks': 40,
        'Bonds': 30,
        'Real Estate': 20,
        'Fixed Deposits': 10
    }

    fig = px.pie(
        values=list(allocations.values()),
        names=list(allocations.keys()),
        title="Recommended Investment Allocation",
        hole=0.3
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="Poppins"),
    )
    fig.update_traces(marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B', '#E0E0E0']))
    return fig