import numpy as np
from scipy.optimize import minimize


def expected_return(weights, returns):
    return np.sum(returns * weights)


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def optimize_portfolio(expected_returns, cov_matrix, risk_tolerance):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(lambda weights, er, cov: -expected_return(weights, er) / portfolio_volatility(weights, cov),
                      num_assets * [1. / num_assets, ], args=args, method='SLSQP', bounds=bounds,
                      constraints=constraints)

    return result.x


def get_asset_allocation(risk_profile, expected_returns, cov_matrix):
    risk_tolerance_map = {
        "Very Conservative": 1,
        "Conservative": 2,
        "Moderate": 3,
        "Growth": 4,
        "Aggressive": 5
    }

    risk_tolerance = risk_tolerance_map.get(risk_profile, 3)
    optimal_weights = optimize_portfolio(expected_returns, cov_matrix, risk_tolerance)

    return dict(zip(['Stocks', 'Bonds', 'Real Estate', 'Cash'], optimal_weights))


def project_wealth(initial_investment, monthly_contribution, years, expected_return, inflation_rate):
    months = years * 12
    monthly_return = (1 + expected_return) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1

    wealth = initial_investment
    for month in range(1, months + 1):
        wealth *= (1 + monthly_return)
        wealth += monthly_contribution
        monthly_contribution *= (1 + monthly_inflation)

    return wealth


def monte_carlo_simulation(initial_investment, monthly_contribution, years, num_simulations=1000):
    results = []
    for _ in range(num_simulations):
        expected_return = np.random.normal(0.07, 0.15)  # Assuming 7% average return with 15% standard deviation
        inflation_rate = np.random.normal(0.02, 0.01)  # Assuming 2% average inflation with 1% standard deviation
        final_wealth = project_wealth(initial_investment, monthly_contribution, years, expected_return, inflation_rate)
        results.append(final_wealth)

    return np.percentile(results, [10, 50, 90])  # Return 10th, 50th, and 90th percentiles


def rebalance_portfolio(current_allocation, target_allocation):
    rebalancing_actions = {}
    for asset, current_pct in current_allocation.items():
        target_pct = target_allocation.get(asset, 0)
        difference = target_pct - current_pct
        if abs(difference) > 0.1:  # Only rebalance if difference is more than 0.1%
            action = "Buy" if difference > 0 else "Sell"
            rebalancing_actions[asset] = f"{action} {abs(difference):.2f}%"
    return rebalancing_actions
