# import streamlit as st
# import pandas as pd
# import joblib
# import sys
# import os


# # Import custom modules
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.append(project_root)
# from src.modeling.predict import make_predictions
# #from src.feature_preprocessing.feature_transformation import log_transformation, encode_categorical_features
# from src.portfolio_optimization.markowitz import calculate_portfolio_statistics, optimize_portfolio
# from src.feature_preprocessing.feature_transformation import (
#     safe_log_transform, log_transformation, encode_categorical_features
# )

# # Load stock data
# stock_data = pd.read_csv(r"D:\Investor_Risk_Tolerance_and_Robo-Advisor\data\raw\stock_data.csv", index_col='Date')


# # Ensure sidebar remains open
# st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# # ---- PAGE TITLE & DESCRIPTION ----
# st.title("📊 Investor Risk Tolerance & Robo-Advisor")

# st.markdown("""
#     This app is a simple robo-advisor that uses a machine learning model to predict the risk tolerance of an investor based on their responses to a questionnaire. The app then uses the predicted risk tolerance to optimize a portfolio allocation using the Markowitz Portfolio Optimization model.
    
#     **📌 Steps to Follow:**
#     - **Step 1:** Enter your investor characteristics to predict your risk tolerance.
#     - **Step 2:** Select your preferred stocks, and the app will generate the optimal portfolio.
# """)

# # ---- SIDEBAR: INVESTOR QUESTIONNAIRE ----
# st.sidebar.header("📝 Step 1: Enter Investor Characteristics")

# def user_input_features():
#     """
#     Collects user input for the questionnaire.
#     """
    
#     age = st.sidebar.number_input("Age", min_value=18, max_value=100,value=25)
#     education_level = st.sidebar.selectbox("Education Level", ['junior_school', 'high_school', 'Bachelors/degree', 'Masters/phd'])
#     marital_status = st.sidebar.selectbox("Marital Status", ['single/divorced', 'married'])
#     no_of_kids = st.sidebar.number_input("Number of Kids", min_value=0, max_value=7, value=2)
#     occupation_category = st.sidebar.selectbox("Occupation Category", ['Unemployed', 'Junior_level', 'mid_level', 'Senior_level',])
#     income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
#     net_worth = st.sidebar.number_input("Net Worth", min_value=0, value=1000000)
#     spending_vs_income = st.sidebar.selectbox("Spending vs Income Ratio", ['spends_more', 'spends_averagely', 'spends_less' ])
#     spending_level = st.sidebar.selectbox("Spending Level (1-5, 5 being high)", [1, 2, 3, 4, 5])
#     risk_level = st.sidebar.slider("willingness to a Risk  (1-4)", 1, 4, 3)
    


#  # **Store input in a dictionary**
#     user_data = {
#     "AGE": age,
#     "EDUCATION_LEVEL": education_level,
#     "MARITAL_STATUS": marital_status,
#     "NO_OF_KIDS": no_of_kids,
#     "OCCUPATION_CATEGORY": occupation_category,
#     "INCOME": income,
#     "RISK_LEVEL": risk_level,
#     "SPENDING_VS_INCOME": spending_vs_income,
#     "SPENDING_LEVEL": spending_level,
#     "NETWORTH": net_worth
    
#    }


#     # **Convert dictionary to a DataFrame and TRANSPOSE it**
#     features = pd.DataFrame(user_data, index=[0])
#     return features 
    

# df = user_input_features()

# # Display user inputs
# st.markdown("###### User Input Summary")
# st.dataframe(df)

# # ---- RISK TOLERANCE PREDICTION ----
# st.markdown("#### 📈 Step 2: Asset Allocation and Portfolio Performance")

# # Load trained model
# model_path = r"D:\Investor_Risk_Tolerance_and_Robo-Advisor\models\best_model.pkl"
# model = joblib.load(model_path)

# # Transform input data
# df = encode_categorical_features(df)

# # Predict risk tolerance
# predicted_risk_tolerance = model.predict(df)[0]
# st.write(f"**Predicted Risk Tolerance Score:** {predicted_risk_tolerance:.2f} (scale of 100)")

# # ---- PORTFOLIO OPTIMIZATION ----
# st.write("##### 📊 Portfolio Optimization")
# selected_stocks = st.multiselect("Select Stocks", stock_data.columns)

# if st.button("SUBMIT"):
#     c = st.container()
        
#     if selected_stocks:
#         sample_data = stock_data[selected_stocks].copy()
#         clean_weights, cumulative_returns = optimize_portfolio(sample_data, predicted_risk_tolerance)
        
        
#         # Create two columns inside the container
#         col1, col2 = c.columns(2)

#         # Bar Chart in the first column
#         with col1:
#             col1.markdown("##### Asset Allocation")
#             col1.bar_chart(clean_weights)

#         # Compute portfolio value assuming an initial investment of $100
#         initial_investment = 100
#         portfolio_value = initial_investment * cumulative_returns

#         # Line Chart in the second column
#         with col2:
#             col2.markdown("##### Portfolio Value of $100 Over Time")
#             col2.line_chart(portfolio_value)

import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.modeling.predict import make_predictions
from src.feature_preprocessing.feature_transformation import safe_log_transform, log_transformation, encode_categorical_features
from src.portfolio_optimization.markowitz import calculate_portfolio_statistics, optimize_portfolio

# Load stock data
stock_data = pd.read_csv(r"D:\Investor_Risk_Tolerance_and_Robo-Advisor\data\raw\stock_data.csv", index_col='Date')

# Ensure sidebar remains open
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ---- PAGE TITLE & DESCRIPTION ----
st.title("📊 Investor Risk Tolerance & Robo-Advisor")
st.markdown("""
This app predicts investor risk tolerance and generates an optimized portfolio.
- Step 1: Enter your characteristics
- Step 2: Select stocks and get portfolio allocation
""")

# ---- SIDEBAR: INVESTOR QUESTIONNAIRE ----
st.sidebar.header("📝 Step 1: Enter Investor Characteristics")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
    education_level = st.sidebar.selectbox("Education Level", ['junior_school', 'high_school', 'Bachelors/degree', 'Masters/phd'])
    marital_status = st.sidebar.selectbox("Marital Status", ['single/divorced', 'married'])
    no_of_kids = st.sidebar.number_input("Number of Kids", min_value=0, max_value=7, value=2)
    occupation_category = st.sidebar.selectbox("Occupation Category", ['Unemployed', 'Junior_level', 'mid_level', 'Senior_level'])
    income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
    net_worth = st.sidebar.number_input("Net Worth", min_value=0, value=1000000)
    spending_vs_income = st.sidebar.selectbox("Spending vs Income Ratio", ['spends_more', 'spends_averagely', 'spends_less'])
    spending_level = st.sidebar.selectbox("Spending Level (1-5, 5 being high)", [1, 2, 3, 4, 5])
    risk_level = st.sidebar.slider("Willingness to Take Risk (1-4)", 1, 4, 3)

    user_data = {
        "AGE": age,
        "EDUCATION_LEVEL": education_level,
        "MARITAL_STATUS": marital_status,
        "NO_OF_KIDS": no_of_kids,
        "OCCUPATION_CATEGORY": occupation_category,
        "INCOME": income,
        "RISK_LEVEL": risk_level,
        "SPENDING_VS_INCOME": spending_vs_income,
        "SPENDING_LEVEL": spending_level,
        "NETWORTH": net_worth
    }

    return pd.DataFrame(user_data, index=[0])

df = user_input_features()
st.markdown("###### User Input Summary")
st.dataframe(df)

# ---- RISK TOLERANCE PREDICTION ----
st.markdown("#### 📈 Step 2: Asset Allocation and Portfolio Performance")

# Apply log transform first
df = safe_log_transform(df, ['INCOME', 'NETWORTH'])
# Encode categorical features
df = encode_categorical_features(df)

# Load trained model
model_path = r"D:\Investor_Risk_Tolerance_and_Robo-Advisor\models\best_model.pkl"
model = joblib.load(model_path)

# Predict risk tolerance
predicted_risk_tolerance = model.predict(df)[0]
st.write(f"**Predicted Risk Tolerance Score:** {predicted_risk_tolerance:.2f} (scale of 100)")

# ---- PORTFOLIO OPTIMIZATION ----
st.write("##### 📊 Portfolio Optimization")
selected_stocks = st.multiselect("Select Stocks", stock_data.columns)

if st.button("SUBMIT"):
    if selected_stocks:
        sample_data = stock_data[selected_stocks].copy()
        clean_weights, cumulative_returns = optimize_portfolio(sample_data, predicted_risk_tolerance)

        col1, col2 = st.columns(2)

        with col1:
            col1.markdown("##### Asset Allocation")
            col1.bar_chart(clean_weights)

        initial_investment = 100
        portfolio_value = initial_investment * cumulative_returns

        with col2:
            col2.markdown("##### Portfolio Value of $100 Over Time")
            col2.line_chart(portfolio_value)

# Steps to fix everything
# 1. Open a terminal in your project folder.
# 2. Activate your environment:
# bash
# conda activate risk tolerance
# 3. Run train.py .
# bash
# python src/modeling/train.py
# • This will retrain multiple models and save the best performing model as best_model.pkl .
# 4. After training finishes, run your Streamlit app:
# bash
# streamlit run app.py
# Copy code
# Copy code
# 6) Copy code