import json
import streamlit as st

def save_user_profile(profile):
    try:
        with open('user_profile.json', 'w') as f:
            json.dump(profile, f)
    except Exception as e:
        st.error(f"An error occurred while saving the user profile: {e}")

def load_user_profile():
    try:
        with open('user_profile.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the user profile: {e}")
        return None

def calculate_risk_score(answers):
    weights = [0.3, 0.25, 0.25, 0.2]  # Weights for each question
    score = sum(int(answer) * weight for answer, weight in zip(answers, weights))

    risk_profiles = {
        (1, 2): "Very Conservative",
        (2, 3): "Conservative",
        (3, 3.5): "Moderate",
        (3.5, 4): "Growth",
        (4, 5): "Aggressive"
    }

    for (lower, upper), profile in risk_profiles.items():
        if lower <= score <= upper:
            return score, profile

    return score, "Unknown"

def get_risk_tolerance(age, investment_horizon, income_stability):
    base_score = 5 - (age / 20)  # Younger investors can take more risk
    horizon_adjustment = min(investment_horizon / 10, 1)  # Longer horizon allows more risk
    income_factor = 1 if income_stability == "High" else 0.8

    risk_tolerance = (base_score + horizon_adjustment) * income_factor
    return min(max(risk_tolerance, 1), 5)  # Ensure score is between 1 and 5