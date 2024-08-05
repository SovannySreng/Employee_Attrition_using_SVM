import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Dropping unnecessary columns
    df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

    # Mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No': 0}
    dict_attrition = {'Yes': 1, 'No': 0}
    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)

    # Creating list of dummy columns
    to_get_dummies_for = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus']

    # Creating dummy variables
    df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)

    # Separating target variable and other variables
    Y = df['Attrition']
    X = df.drop(columns=['Attrition'])

    # Scaling the data
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, Y