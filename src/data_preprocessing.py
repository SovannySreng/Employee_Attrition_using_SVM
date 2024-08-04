import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def preprocess_data(df: pd.DataFrame):
    df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
    cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']
    
    Y = df.Attrition.map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Attrition'])
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)