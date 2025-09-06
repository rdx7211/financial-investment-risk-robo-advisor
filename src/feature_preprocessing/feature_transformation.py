# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder



# df = pd.read_pickle(r'D:\Investor_Risk_Tolerance_and_Robo-Advisor\data\interim\processed_data.pkl')

# df 



# def log_transformation(df, columns_to_log_transform):
#     """
#     Applies log transformation to specified columns.
#     """
#     for col in columns_to_log_transform:
#         df[col] = np.log1p(df[col])  

#     return df

# df = log_transformation(df, ['INCOME', 'NETWORTH'])


# def encode_categorical_features(df):
#     """
#     Encodes categorical features using Label Encoding.
#     """
#     le = LabelEncoder()
#     df['EDUCATION_LEVEL'] = le.fit_transform(df['EDUCATION_LEVEL'])
#     df['MARITAL_STATUS'] = le.fit_transform(df['MARITAL_STATUS'])
#     df['OCCUPATION_CATEGORY'] = le.fit_transform(df['OCCUPATION_CATEGORY'])
#     df['SPENDING_VS_INCOME'] = le.fit_transform(df['SPENDING_LEVEL'])
#     return df

# df = encode_categorical_features(df)



# df.to_pickle(r'D:\Investor_Risk_Tolerance_and_Robo-Advisor\data\processed\processed_data.pkl')
# src/feature_preprocessing/feature_transformation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def safe_log_transform(X, columns=None):
    """
    Apply log1p safely to numeric columns.
    Can handle DataFrame or numpy array.
    """
    if isinstance(X, pd.DataFrame):
        X_copy = X.copy()
        if columns is None:
            columns = X_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in columns:
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy
    else:
        # If it's a numpy array, apply log1p element-wise
        return np.log1p(X)

def log_transformation(df, columns_to_log_transform):
    """
    Applies log1p transformation to specified columns in a DataFrame.
    """
    return safe_log_transform(df, columns_to_log_transform)

def encode_categorical_features(df):
    """
    Encodes categorical features using Label Encoding.
    """
    df_copy = df.copy()
    le = LabelEncoder()

    cat_cols = ['EDUCATION_LEVEL', 'MARITAL_STATUS', 'OCCUPATION_CATEGORY', 'SPENDING_VS_INCOME']
    for col in cat_cols:
        if col in df_copy.columns:
            df_copy[col] = le.fit_transform(df_copy[col])
    
    return df_copy
