import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle(r"D:\\Investor_Risk_Tolerance_and_Robo-Advisor\\data\\interim\\processed_data.pkl")

df



def different_features(df):
    """
    Identifies discrete and continuous features in the DataFrame.
    """
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']
    continuous_features = [col for col in df.columns if (df[col].dtype == 'int' or df[col].dtype == 'float') and len(df[col].unique()) >= 10 and col != 'RISK_TOLERANCE']
    discrete_features = [col for col in df.columns if col not in categorical_features and col not in continuous_features and len(df[col].unique()) < 10]

    return categorical_features, discrete_features, continuous_features

categorical_features, discrete_features, continuous_features = different_features(df)

categorical_features, discrete_features, continuous_features

def plot_features(df,categorical_features, discrete_features, continuous_features):
    """
    Plots count plots for discrete features and distribution plots for continuous features.
    """
    # Plot count plots for discrete features
    for col in discrete_features + categorical_features:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette="viridis")
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.show()
    
    # Plot distribution plots for continuous features
    for col in continuous_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution Plot of {col}')
        plt.show()


plot_features(df,categorical_features, discrete_features, continuous_features)