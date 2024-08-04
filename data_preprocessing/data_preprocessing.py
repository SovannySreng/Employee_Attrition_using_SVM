import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path='H:\\My Drive\\BISI II\\Data Science\\Term Assignments\\Employee_Attrition_using_SVM\\data\\HR_Employee_Attrition.xlsx'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    # Convert categorical variables to numeric
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)